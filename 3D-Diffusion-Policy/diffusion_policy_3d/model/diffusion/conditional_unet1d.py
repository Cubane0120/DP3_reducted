from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from termcolor import cprint
from diffusion_policy_3d.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy_3d.common.model_util import print_params
import numpy as np
import os
import pathlib

logger = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)   # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]


        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, horizon]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]
        
        return attn_output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConditionalResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        
        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == 'film': # FiLM modulation https://arxiv.org/abs/1709.07871
            # predicts per-channel scale and bias
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")
        
        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)  
        if cond is not None:      
            if self.condition_type == 'film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1) # [batch_size, out_channels, horizon]
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)
                embed = embed.permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

    
    
class ProjectionModule(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        using_SVD_init=False,
        path_basis=None,
        is_trainable=False,
        is_trainable_partially=False,
        grad_scale=1e-2,
        alpha=0.1,
        ):
        super().__init__()
        
        self.projector = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        if is_trainable and is_trainable_partially:
            raise ValueError("Both is_trainable and is_trainable_partially can't be True")
        
        self.is_trainable_partially = is_trainable_partially
        self.is_trainable = is_trainable
        
        if using_SVD_init:
            if not os.path.isfile(path_basis):
                raise FileNotFoundError(f"{path_basis} not found")
            
            projection_SVD_np = np.load(path_basis)
            projection_SVD_np_k = projection_SVD_np[:, :out_channels]
            projection_SVD_t  = torch.from_numpy(projection_SVD_np_k).to(self.projector.weight.dtype)
            w = projection_SVD_t.t().unsqueeze(-1).contiguous()
            if w.shape != self.projector.weight.shape:
                raise ValueError(f"SVD shape mismatch: got {tuple(w.shape)}, "
                                 f"expected {tuple(self.projector.weight.shape)}")
            
            with torch.no_grad():
                self.projector.weight.copy_(w)
        else:
            if not is_trainable:
                raise ValueError(f"freezing projection module without initial parameters")

        if is_trainable_partially:
            self.delta = nn.Parameter(torch.zeros_like(self.projector.weight))
            self.alpha = float(alpha)

            if grad_scale != 1.0:
                self.delta.register_hook(lambda g: g * grad_scale)
            self.register_buffer("W0_base", self.projector.weight.detach().clone(), persistent=True)
        else:
            self.delta = None
            self.register_buffer("W0_base", None, persistent=False)
        
        if not is_trainable:
            self.projector.weight.requires_grad_(False)
            if is_trainable_partially:
                print(f"projection module C : {in_channels} -> {out_channels} is trainable partially")
            else:
                print(f"projection module C : {in_channels} -> {out_channels} is frozen")
        else:
            print(f"projection module C : {in_channels} -> {out_channels} is trainable")


    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.is_trainable_partially:
            with torch.no_grad():
                self.projector.weight.copy_(self.W0_base)
                
        if not mode and self.is_trainable_partially:
            with torch.no_grad():
                fused = self.W0_base * (1.0 + self.alpha * torch.tanh(self.delta))
                self.projector.weight.copy_(fused)
        return self

    def forward(self, x):
        if not self.training:
            return self.projector(x)
        
        if self.is_trainable_partially:
            W  = self.W0_base * (1.0 + self.alpha * torch.tanh(self.delta))
            
            return F.conv1d(
                x, W, bias=None,
                stride=self.projector.stride,
                padding=self.projector.padding,
                dilation=self.projector.dilation,
                groups=self.projector.groups,
            )
        else:
            return self.projector(x)

def Select_dimension_with_PercentVariance(path, percentVar, n):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    
    normalized_singular_values = np.load(path)
    cumulative_variance = np.cumsum(normalized_singular_values)
    dim = np.searchsorted(cumulative_variance, percentVar)
    dim = int(np.ceil(dim / n) * n)
    return dim

def Get_PercentVariance(path, dim):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    normalized_singular_values = np.load(path)
    cumulative_variance = np.cumsum(normalized_singular_values)
    percent_variance = cumulative_variance[dim]
    return percent_variance*100

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        # collecting output Tensor to calculate SVD basis
        collect_outputTensor=False, #aadsad
        collect_outputTensor_path=None,
        sampling_type=None,
        # projection module condition
        using_projection=False,
        percentVar = None,
        projection_h1_dim = None,
        projection_h2_dim = None,
        projection_reduction_rate = 2,
        
        # SVD's projection
        using_projection_SVD_init=False,
        path_basis=None,
        
        freezing_early_module=False,
        freezing_diffusion_step_encoder=False,
        pojection_is_trainable=False,
        pojection_is_trainable_partially=False,
        ):
        
        super().__init__()
        
        self.condition_type = condition_type
        
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        
        
        self.collect_outputTensor = collect_outputTensor
        if collect_outputTensor:
            self.collect_outputTensor_path = collect_outputTensor_path
            self.idx_save_h1 = 0
            self.idx_save_h2 = 0
        self.using_projection = using_projection
        

        if sampling_type is not None:
            sampling_type = str(sampling_type)
            
            if sampling_type not in ['uniform', '2-anchor', '1-anchor', 'hybrid']:
                raise ValueError(f"sampling_type {sampling_type} not supported")
        else:
            if self.using_projection:
                raise ValueError("sampling_type must be specified when using_projection is True")
        
        if self.using_projection:
            if not os.path.isdir(path_basis):
                raise FileNotFoundError(f"{path_basis} not found")
            
            path_basis = pathlib.Path(path_basis)
            path_basis_h1 = path_basis / (sampling_type + "_V_latent_h1.npy")
            path_basis_h2 = path_basis / (sampling_type + "_V_latent_h2.npy")
            
            path_basis_S1 = path_basis / (sampling_type + "_S_latent_h1.npy")
            path_basis_S2 = path_basis / (sampling_type + "_S_latent_h2.npy")
            
            if percentVar is not None:
                if not isinstance(percentVar, float) or not (percentVar > 0. and percentVar < 1.):
                    raise ValueError("Percent variance is not float in (0, 1.)")
                if (projection_h1_dim is not None) or (projection_h2_dim is not None):
                    raise ValueError("Using Percent variance reference, but dim is assigned")
                
                projection_h1_dim, projection_h2_dim = \
                    Select_dimension_with_PercentVariance(path_basis_S1, percentVar, n_groups),\
                    Select_dimension_with_PercentVariance(path_basis_S2, percentVar, n_groups)
            
            P_S1, P_S2 = \
                Get_PercentVariance(path_basis_S1, projection_h1_dim), \
                Get_PercentVariance(path_basis_S2, projection_h2_dim)
            print(f"k1: {projection_h1_dim}")
            print(f"k2: {projection_h2_dim}")
            print(f"P_S1: {P_S1}")
            print(f"P_S2: {P_S2}")

            self.projection_h1 = ProjectionModule(
                in_channels=down_dims[1], 
                out_channels=projection_h1_dim,
                using_SVD_init=using_projection_SVD_init,
                path_basis=path_basis_h1,
                is_trainable=pojection_is_trainable,
                is_trainable_partially=pojection_is_trainable_partially,
            )
            self.projection_h2 = ProjectionModule(
                in_channels=down_dims[2], 
                out_channels=projection_h2_dim,
                using_SVD_init=using_projection_SVD_init,
                path_basis=path_basis_h2,
                is_trainable=pojection_is_trainable,
                is_trainable_partially=pojection_is_trainable_partially,
            )
            
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type)
            ])
        
        if self.using_projection:
            mid_dim = projection_h2_dim
        else:
            mid_dim = all_dims[-1]
            
        
        UnitBlock = ConditionalResidualBlock1D

        self.mid_modules = nn.ModuleList([
            UnitBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            UnitBlock(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                UnitBlock(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                UnitBlock(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])

        # breakpoint()
        if not self.using_projection:
            out_in = reversed(in_out[1:])

            for ind, (dim_in, dim_out) in enumerate(out_in):
                is_last = ind >= (len(in_out) - 1)
                up_modules.append(nn.ModuleList([
                    UnitBlock(
                        dim_out*2, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type),
                    UnitBlock(
                        dim_in, dim_in, cond_dim=cond_dim,
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))
            
            final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
                nn.Conv1d(start_dim, input_dim, 1),
            )
        else:
            up_1_in = 2 * projection_h2_dim
            up_1_out = (up_1_in//projection_reduction_rate)//n_groups * n_groups
            
            up_2_in = projection_h1_dim + up_1_out
            up_2_out = (up_2_in//projection_reduction_rate)//n_groups * n_groups
            
            out_in = (
                [up_1_in, up_1_out],
                [up_2_in, up_2_out],
            )

            for ind, (dim_in, dim_out) in enumerate(out_in):
                is_last = ind >= (len(in_out) - 1)
                up_modules.append(nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type),
                    ConditionalResidualBlock1D(
                        dim_out, dim_out, cond_dim=cond_dim, 
                        kernel_size=kernel_size, n_groups=n_groups,
                        condition_type=condition_type),
                    Upsample1d(dim_out) if not is_last else nn.Identity()
                ]))

            f_dim = out_in[-1][-1]
            final_conv = nn.Sequential(
                Conv1dBlock(f_dim, f_dim, kernel_size=kernel_size),
                nn.Conv1d(f_dim, input_dim, 1),
            )
        # breakpoint()

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        print_params(self)

        print(f"num_param: {sum(p.numel() for p in self.parameters())}")

        if self.using_projection:
            if using_projection_SVD_init:
                print(f"[ConditionalUnet1D] using path basis h1: %s", path_basis_h1)
                print(f"[ConditionalUnet1D] using path basis h2: %s", path_basis_h2)

        
        if freezing_early_module:
            for p in self.down_modules.parameters():
                p.requires_grad = False
            print(f"freezing down_modules")
        if freezing_diffusion_step_encoder:
            for p in self.diffusion_step_encoder.parameters():
                p.requires_grad = False
            print(f"freezing diffusion step encoder")

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """

        #breakpoint()
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            if self.condition_type == 'cross_attention':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_feature = torch.cat([timestep_embed, global_cond], axis=-1)


        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        # breakpoint()
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            
            if self.using_projection:
                if idx == 1:
                    x_ = self.projection_h1(x)
                    h.append(x_)
                elif idx == 2:
                    x = self.projection_h2(x)
                    h.append(x)
                else:
                    h.append(x)
            else:
                h.append(x)
            if self.collect_outputTensor and idx!=0:
                # breakpoint()
                # save bottleneck_out
                latent = x.detach().cpu().numpy()  # (1,1024,8)/(1,2048,4)
                
                #@ edit for collect in train data
                #latent = latent.squeeze(0)
                
                #np.save(f"/home/hsh/3D-Diffusion-Policy/data_bottleneck_out/latent_sample_{self.idx_save}.npy", latent)
                if self.collect_outputTensor_path is None:
                    raise ValueError("collect_data_path must be specified when collect_data is True")
                if not os.path.exists(self.collect_outputTensor_path):
                    os.makedirs(self.collect_outputTensor_path)
                
                if idx == 1:
                    np.save(f"{self.collect_outputTensor_path}/latent_h1_{self.idx_save_h1}.npy", latent)
                    self.idx_save_h1+=1
                elif idx == 2:
                    np.save(f"{self.collect_outputTensor_path}/latent_h2_{self.idx_save_h2}.npy", latent)
                    self.idx_save_h2+=1
                else:
                    raise ValueError(f"idx {idx} in down_modules is not supported for collect_data")
            x = downsample(x)
            # breakpoint()

        if self.collect_outputTensor:
            return (self.idx_save_h1 == self.idx_save_h2)
        
        # breakpoint()
        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)
            

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)

            if self.use_up_condition: 
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)
            # breakpoint()    
        # breakpoint()

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        # breakpoint()
        return x