if __name__ == "__main__":
    import sys
    import os
    import pathlib

    os.environ["OMP_NUM_THREADS"] = "32"
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import hydra
from omegaconf import OmegaConf
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import cupy as cp
import pathlib
import logging

logger = logging.getLogger(__name__)

# prefix = "latent_h1"
# prefix = "latent_h2"
# prefix = "latent_md"
prefixes = ["latent_h1", "latent_h2"]#, "latent_md"]

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    data_path = cfg.policy.collect_data_path
    k_val = [cfg.k1, cfg.k2]  # number of components to keep
    for i_prefix, prefix in enumerate(prefixes):
        print("Processing prefix:", prefix)
        file_paths = sorted(glob.glob(data_path + f"/{prefix}*.npy"))
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_path}")
        else:
            print(f"Found {len(file_paths)} files in {data_path}")

        all_latents = [np.load(f) for f in file_paths]  # list of (1024, 8) / (2048, 4)
        # breakpoint()
        #@ when output tensors from train data(batch not 1)
        all_latents = np.concatenate([np.asarray(b)[None, ...] for row in all_latents for b in row],axis=0)
        # all_latents = np.stack(all_latents)  # shape: (N, 1024, 8) / (N, 2048, 4)
        print(f"Shape of all_latents: {all_latents.shape}")
        dim_out = all_latents.shape[1]  # 1024 or 2048

        # 1) (N, 2048, 4) → (N, 4, 2048)
        all_latents = all_latents.transpose(0, 2, 1)
        # 2) (N, 4, 2048) → (N*4, 2048)
        X = all_latents.reshape(-1, dim_out)
        X_gpu = cp.asarray(X)
        #U, S, VT = np.linalg.svd(X, full_matrices=False)  # S: (min(N, D),)
        U, S, VT = cp.linalg.svd(X_gpu, full_matrices=False)

        print(S.shape)
        # 4. Plot explained variance ratio
        # explained_variance_ratio = S**2 / np.sum(S**2)
        # accumulate_var_ratio = np.cumsum(explained_variance_ratio)
        # idx_list = []
        # for threshold in [0.97, 0.98, 0.99, 0.999]:
        #     idx = np.searchsorted(accumulate_var_ratio, threshold, side='left')
        #     idx_list.append(idx)
        # [163, 220, 342, 895]

        # #plt.plot(accumulate_var_ratio, marker='o')
        # plt.plot(accumulate_var_ratio[:300], marker='o')
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cumulative Explained Variance')
        # plt.title('SVD Component Importance')
        # plt.grid()
        # plt.show()

        # 5. Truncate to top-K components
        #explained_variance_ratio = S**2 / np.sum(S**2)
        explained_variance_ratio = S**2 / cp.sum(S**2)
        #accumulate_var_ratio = np.cumsum(explained_variance_ratio)
        accumulate_var_ratio = cp.cumsum(explained_variance_ratio)

        data_dir = pathlib.Path(data_path).parent / "basis"
        data_dir.mkdir(parents=True, exist_ok=True)

        #k = np.searchsorted(accumulate_var_ratio, threshold, side='left')
        k = k_val[i_prefix]
        acc_var = accumulate_var_ratio[k-1].item()
        logger.info(
            f"prefixes : {prefix}, Number of components to retain {acc_var*100}% variance: {k}"
        )
        #{str(threshold)}
        VT_k = VT[:k, :]              # (k, D)
        V_k = VT_k.T                  # (D, k) 
        V_k_cpu = cp.asnumpy(V_k)
        
        save_dir = data_dir / f"threshold_fixed"
        save_dir.mkdir(parents=True, exist_ok=True)
        #np.save(data_path+f"_svd_basis_{prefix}.npy", V_k)  # shape: (D, k)        
        np.save(str(save_dir)+f"/{prefix}.npy", V_k_cpu)  # shape: (D, k)

if __name__ == "__main__":
    main()

