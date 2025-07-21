if __name__ == "__main__":
    import sys
    import os
    import pathlib

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



prefix = "latent_h1"
data_path = "3D-Diffusion-Policy/data/outputs/metaworld_disassemble-dp3-0710_seed0/collect_data"

print("Processing prefix:", prefix)

file_paths = sorted(glob.glob(data_path + f"/{prefix}*.npy"))
if len(file_paths) == 0:
    raise FileNotFoundError(f"No .npy files found in {data_path}")
else:
    print(f"Found {len(file_paths)} files in {data_path}")

all_latents = [np.load(f) for f in file_paths]  # list of (1024, 8) / (2048, 4)
all_latents = np.stack(all_latents)  # shape: (N, 1024, 8) / (N, 2048, 4)
print(f"Shape of all_latents: {all_latents.shape}")
dim_out = all_latents.shape[1]  # 1024 or 2048

all_latents = all_latents.transpose(0, 2, 1)
X = all_latents.reshape(-1, dim_out)

U, S, VT = np.linalg.svd(X, full_matrices=False)  # S: (min(N, D),)


explained_variance_ratio = S**2 / np.sum(S**2)
accumulate_var_ratio = np.cumsum(explained_variance_ratio)
idx_list = []
threshold = 0.99
k = np.searchsorted(accumulate_var_ratio, threshold, side='left')
print(f"Number of components to retain {threshold*100}% variance: {k}")

U_k = U[:, :k]                # (N, k)
S_k = S[:k]                   # (k,)
VT_k = VT[:k, :]              # (k, D)
V_k = VT_k.T                  # (D, k) 

weights = V_k
th = 0.01
while True:
    breakpoint()
    useless_in = np.where(np.all(abs(weights) < th, axis=0))[0]

    # 2) 완전 0인 출력 채널 인덱스 (만약 k 중에 있는 경우)
    useless_out = np.where(np.all(abs(weights) < th, axis=1))[0]

    print("입력 채널 중 기여 없는 인덱스:", useless_in)
    print("출력 채널 중 기여 없는 인덱스:", useless_out)