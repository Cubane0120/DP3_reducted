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
    data_path = cfg.policy.collect_outputTensor_path
    save_dir = pathlib.Path(data_path).parent / "basis"
    save_dir.mkdir(parents=True, exist_ok=True)

    sampling_type = cfg.policy.sampling_type
    if sampling_type is None:
        raise ValueError("sampling_type must be specified in the config file")
    else:
        sampling_type = str(sampling_type)    
        if sampling_type not in ['1-anchor', 'uniform', 'linear', 'central']:
            raise ValueError(f"sampling_type {sampling_type} not supported")

    for i_prefix, prefix in enumerate(prefixes):
        print("Processing prefix:", prefix)
        file_paths = sorted(glob.glob(data_path + f"/{prefix}*.npy"))
        if len(file_paths) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_path}")
        else:
            print(f"Found {len(file_paths)} files in {data_path}")

        all_latents = [np.load(f) for f in file_paths]  # list of (1024, 8) / (2048, 4)
        all_latents = np.concatenate([np.asarray(b)[None, ...] for row in all_latents for b in row],axis=0)

        print(f"Shape of all_latents: {all_latents.shape}")
        dim_out = all_latents.shape[1]  # 1024 or 2048

        # 1) (N, C, 4) → (N, 4, C)
        all_latents = all_latents.transpose(0, 2, 1)
        # 2) (N, 4, C) → (N*4, 2048)
        X = all_latents.reshape(-1, dim_out)
        X_gpu = cp.asarray(X)
        U, s, VT = cp.linalg.svd(X_gpu, full_matrices=False)
        S = s**2 / (s**2).sum()  # normalized singular values
        
        path = "/SSDa/dongwoo_nam/hsh/DP3_reducted/" + sampling_type + "_" + str(cfg.name) + "_" + prefix + ".csv"
        s_cpu = cp.asnumpy(s).ravel()   # -> numpy 1D
        S_cpu = cp.asnumpy(S).ravel()   # -> numpy 1D
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(cfg.task_name) + ",")
            f.write(",".join(map(str, s_cpu)) + "\n")
        np.save(str(save_dir)+f"/{sampling_type}_S_{prefix}.npy", S_cpu)  # shape: (D)

        V = VT.T                  # (D, k) 
        V_cpu = cp.asnumpy(V)
        np.save(str(save_dir)+f"/{sampling_type}_V_{prefix}.npy", V_cpu)  # shape: (D, k)

if __name__ == "__main__":
    main()

