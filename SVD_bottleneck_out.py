import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# 1. Load all latent vectors
file_paths = sorted(glob.glob("/home/hsh/3D-Diffusion-Policy/data_bottleneck_out/*.npy"))
all_latents = [np.load(f) for f in file_paths]  # list of (2048, 4)
all_latents = np.stack(all_latents)  # shape: (N, 2048, 4)

# 2. Reshape to 2D (samples, features)

X = all_latents.reshape(len(all_latents), -1)

# 3. Full SVD for importance analysis
U, S, VT = np.linalg.svd(X, full_matrices=False)  # S: (min(N, D),)

# 4. Plot explained variance ratio
explained_variance_ratio = S**2 / np.sum(S**2)
accumulate_var_ratio = np.cumsum(explained_variance_ratio)

#plt.plot(accumulate_var_ratio, marker='o')
plt.plot(accumulate_var_ratio[:200], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('SVD Component Importance')
plt.grid()
plt.show()

# 40 : 0.97
# 47 : 0.98
# 59 : 0.99
# 73 : 0.995
# 124: 0.999

# 5. Truncate to top-K components
k = 59  # or whatever you decide from plot
svd = TruncatedSVD(n_components=k)
X_reduced = svd.fit_transform(X)  # shape: (N, k)

# Optional: save truncated basis
np.save("latent_svd_basis.npy", svd.components_)  # shape: (k, D)