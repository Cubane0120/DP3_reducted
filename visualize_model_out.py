import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from mpl_toolkits.mplot3d import Axes3D

# 불러오기
with open("3D-Diffusion-Policy/noise_gradient_data.pkl", "rb") as f:
    data = pickle.load(f)

noise_vals = data["noise_vals"]
grid_x = data["grid_x"]
grid_y = data["grid_y"]
c_idx_a = data["c_idx_a"]
c_idx_b = data["c_idx_b"]
t_idx = data["t_idx"]

# reshape
noise_array = np.array(noise_vals)
U = noise_array[:, :, 0]
V = noise_array[:, :, 1]

dx = grid_x[1, 0] - grid_x[0, 0]
dy = grid_y[0, 1] - grid_y[0, 0]


# U = ∂E/∂x, V = ∂E/∂y 라고 가정하면
# energy_from_U: x 축(행, axis=0) 으로 통합
energy_from_U = cumtrapz(-U, dx=dx, axis=0, initial=0)
# energy_from_V: y 축(열, axis=1) 으로 통합
energy_from_V = cumtrapz(-V, dx=dy, axis=1, initial=0)

# 두 결과를 합치면 potential (up to 상수) 가 복원됩니다.
energy_estimate = energy_from_U + energy_from_V
breakpoint()
# 3D 시각화
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid_x, grid_y, energy_estimate, cmap='viridis', edgecolor='none')
ax.set_title('3D Energy Landscape')
ax.set_xlabel(f'Perturbation at chan={c_idx_a}')
ax.set_ylabel(f'Perturbation at chan={c_idx_b}')
ax.set_zlabel('Energy')
plt.tight_layout()
plt.show()
