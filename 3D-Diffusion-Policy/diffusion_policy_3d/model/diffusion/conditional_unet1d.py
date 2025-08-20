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

from mamba_ssm.modules.mamba_simple import Mamba
# try:
#     from diffusion_policy_3d.model.diffusion.hydra_ssm import Hydra
# except ImportError:
#     Hydra = None

# try:
#     from mamba_ssm.modules.mamba2 import Mamba2
# except ImportError:
#     Mamba2 = None
# # from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath

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

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 


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

class MambaVisionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.2, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 mamba_type='v1',  ## ['v1', 'v2', 'bi', 'hydra']
                 ):
        super().__init__()

        cprint(f'Using Mamba {mamba_type} version', 'yellow')
        self.norm1 = norm_layer(dim)
        if mamba_type == 'v1':
            self.mixer = Mamba(d_model=dim)
        elif mamba_type == 'v2':
            self.mixer = Mamba2(d_model=dim, headdim=32)
        elif mamba_type == 'bi':
            self.mixer = Mamba(d_model=dim, bimamba=True)
        elif mamba_type == 'hydra':
            self.mixer = Hydra(d_model=dim)
        else:
            NotImplementedError(f"mamba_type {mamba_type} not implemented")
                 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1)

        return x

class AttnVisionBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8,
                 qkv_bias=False, 
                 qk_scale=False, 
                 attn_drop=0.,
                 mlp_ratio=4., 
                 drop=0., 
                 drop_path=0.2, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
                        dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        norm_layer=norm_layer,
                    )
                 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1)

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


class ConditionalMambaResidualBlock1D(nn.Module):
    mamba_version='mambavision_v1'

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film',
                 ):
        super().__init__()
        mamba_version = self.mamba_version

        if 'mambavision' in mamba_version:
            if 'v1' in mamba_version:
                mamba_type = 'v1'
            elif 'v2' in mamba_version:
                mamba_type = 'v2'
            elif 'bi' in mamba_version:
                mamba_type = 'bi'
            elif 'hydra' in mamba_version:
                mamba_type = 'hydra'
            else:
                NotImplementedError(f"mamba_version {mamba_version} not implemented")
            self.blocks = nn.ModuleList([
                Conv1dBlock(in_channels,
                            out_channels,
                            kernel_size,
                            n_groups=n_groups),
                MambaVisionBlock(out_channels, mamba_type=mamba_type),
                AttnVisionBlock(out_channels,),
            ])
        else:
            raise NotImplementedError(f"mamba_version {mamba_version} not implemented")
        
        self.mamba_version = mamba_version
        
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

    def forward(self, x, cond=None, time_cond=None):
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

        if self.mamba_version == 'mambavision':
            out = self.blocks[2](out)

        out = out + self.residual_conv(x)

        return out

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
        collect_data=False,
        collect_data_path=None,
        path_basis_h1=None,
        path_basis_h2=None,
        mamba_version=None, #'mambavision_v1',
        ):
        super().__init__()
        if path_basis_h1 is not None:
            if not os.path.isfile(path_basis_h1):
                raise FileNotFoundError(f"{path_basis_h1} not found")
            # reductor_np = np.load(path_basis)
            # reductor_t = torch.from_numpy(reductor_np)
            # #self.reductor= self.reductor.float()   
            # self.register_buffer("reductor", reductor_t) 

            reductor_np = np.load(path_basis_h1)
            reductor_t  = torch.from_numpy(reductor_np)
            # Conv1d 모듈 생성
            self.reductor_h1_conv = nn.Conv1d(
                in_channels= reductor_t.shape[0], 
                out_channels=reductor_t.shape[1],
                kernel_size=1,
                bias=False
            )
            with torch.no_grad():
                self.reductor_h1_conv.weight.copy_(reductor_t.t().unsqueeze(-1))
            k_h1 = reductor_t.shape[1]
            print(f"k_h1: {k_h1}")
        else:
            k_h1 = down_dims[1]
        if path_basis_h2 is not None:
            if not os.path.isfile(path_basis_h2):
                raise FileNotFoundError(f"{path_basis_h2} not found")
            # reductor_np = np.load(path_basis)
            # reductor_t = torch.from_numpy(reductor_np)
            # #self.reductor= self.reductor.float()   
            # self.register_buffer("reductor", reductor_t) 

            reductor_np = np.load(path_basis_h2)
            reductor_t  = torch.from_numpy(reductor_np)
            # Conv1d 모듈 생성
            self.reductor_h2_conv = nn.Conv1d(
                in_channels= reductor_t.shape[0], 
                out_channels=reductor_t.shape[1],
                kernel_size=1,
                bias=False
            )
            with torch.no_grad():
                self.reductor_h2_conv.weight.copy_(reductor_t.t().unsqueeze(-1))
            k_h2 = reductor_t.shape[1]
            print(f"k_h2: {k_h2}")
        else:
            k_h2 = down_dims[2]

        self.path_basis_h1 = path_basis_h1
        self.path_basis_h2 = path_basis_h2

        self.condition_type = condition_type
        
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        self.collect_data = collect_data
        if collect_data:
            self.collect_data_path = collect_data_path
            self.idx_save_h1 = 0
            self.idx_save_h2 = 0

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
        
        # mid_dim = all_dims[-1]
        
        #ver2
        if self.path_basis_h2 is None:
            mid_dim = all_dims[-1]
        else:
            mid_dim = k_h2
        
        #about mamba or not
        if mamba_version:
            UnitBlock = ConditionalMambaResidualBlock1D
            UnitBlock.mamba_version = mamba_version
        else:
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
        if path_basis_h1 is None and path_basis_h2 is None:
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
            out_in = (
                [  
                    2 * k_h2,
                    k_h2
                ],
                [
                    k_h1 + k_h2,
                    ( ((k_h1 + k_h2)//2-1)//8 + 1 ) * 8
                ] 
            )
            # print(out_in)
            # print(k_h1, k_h2)
            # breakpoint()
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

        if path_basis_h1 is not None or path_basis_h2 is not None:
            for p in self.parameters():
                p.requires_grad = False

            for p in self.mid_modules.parameters():
                p.requires_grad = True
            for p in self.up_modules.parameters():
                p.requires_grad = True
            for p in self.final_conv.parameters():
                p.requires_grad = True

            
            
            print(f"[ConditionalUnet1D] using path basis h1: %s", path_basis_h1)
            print(f"[ConditionalUnet1D] using path basis h2: %s", path_basis_h2)
            print(f"freezing model without up_modules and final_conv")

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
            
            # ver 2
            if self.path_basis_h1 is not None and idx == 1:
                x_ = self.reductor_h1_conv(x)
                h.append(x_)
            elif self.path_basis_h2 is not None and idx == 2:
                x= self.reductor_h2_conv(x)
                h.append(x)
            else:
                h.append(x)
            if self.collect_data and idx!=0:
                # breakpoint()
                # save bottleneck_out
                latent = x.detach().cpu().numpy()  # (1,1024,8)/(1,2048,4)
                
                #@ edit for collect in train data
                #latent = latent.squeeze(0)
                
                #np.save(f"/home/hsh/3D-Diffusion-Policy/data_bottleneck_out/latent_sample_{self.idx_save}.npy", latent)
                if self.collect_data_path is None:
                    raise ValueError("collect_data_path must be specified when collect_data is True")
                if not os.path.exists(self.collect_data_path):
                    os.makedirs(self.collect_data_path)
                
                if idx == 1:
                    np.save(f"{self.collect_data_path}/latent_h1_{self.idx_save_h1}.npy", latent)
                    self.idx_save_h1+=1
                elif idx == 2:
                    np.save(f"{self.collect_data_path}/latent_h2_{self.idx_save_h2}.npy", latent)
                    self.idx_save_h2+=1
                else:
                    raise ValueError(f"idx {idx} in down_modules is not supported for collect_data")
            x = downsample(x)
            # breakpoint()

        # breakpoint()
        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)
            

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # if self.path_basis_h2 is not None and idx == 0:
            #     h_reducted = self.reductor_h2_conv(h.pop())
            #     x = torch.cat((x, h_reducted), dim=1)
            # elif self.path_basis_h1 is not None and idx == 1:
            #     h_reducted = self.reductor_h1_conv(h.pop())
            #     x = torch.cat((x, h_reducted), dim=1)
            # else:
            #     x = torch.cat((x, h.pop()), dim=1)

            #ver 2
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