#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import matplotlib.gridspec as gridspec

# Load data

# Paths (adjust if needed)
test_h5 = 'test.h5'
obs_h5 = 'obs.h5'

print(f"Loading test trajectories from: {test_h5}")
print(f"Loading observations from: {obs_h5}")

# Load ONE full trajectory (sample index 0)
with h5py.File(test_h5, 'r') as f:
    x_true_full = f['x'][0, :, :]  # Shape: (1024, 3)
    x_true_all = x_true_full[:120, :]  # Keep only first 120 timesteps for visualization
    
with h5py.File(obs_h5, 'r') as f:
    y_obs = f['lo'][0, :]  # Shape: (9, 1) - low frequency observations
    
#  Create figure with GridSpec (5 subplots)

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

ax_3d = fig.add_subplot(gs[0, 0], projection='3d')

# Plot full trajectory in 3D
x, y, z = x_true_all[:, 0], x_true_all[:, 1], x_true_all[:, 2]

colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
ax_3d.plot(x, y, z, color='lightgray', alpha=0.5, linewidth=1, zorder=1)

scatter = ax_3d.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', 
                        s=5, alpha=0.7, zorder=2)

ax_3d.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, marker='o', 
              label='Start', zorder=3)
ax_3d.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100, marker='^', 
              label='End', zorder=3)

ax_3d.set_xlabel('$x$', fontsize=10)
ax_3d.set_ylabel('$y$', fontsize=10)
ax_3d.set_zlabel('$z$', fontsize=10)
ax_3d.set_title('Hidden State: $\mathbf{x}_{1:L}$ (3D)', 
                fontsize=11, fontweight='bold')
ax_3d.legend(loc='upper left', fontsize=9)

ax_obs = fig.add_subplot(gs[0, 1:])

time_all = np.arange(120)
obs_indices = np.arange(0, 65, 8)  # [0, 8, 16, 24, 32, 40, 48, 56, 64] = 9 points
obs_times = obs_indices.astype(float)
obs_values = y_obs.flatten()  # Shape: (9,)

ax_obs.plot(time_all, x_true_all[:, 0], color='black', alpha=0.7, 
            linewidth=2.5, label='True $x(t)$ (hidden)', zorder=1)

noise_std = 0.05  # Observation noise level
ax_obs.errorbar(obs_times, obs_values, yerr=noise_std, fmt='s', 
                color='darkblue', ecolor='darkblue', alpha=0.8, markersize=8, 
                elinewidth=2, capsize=4, label='Observations $y_i$ (sparse + clean)', 
                zorder=3)

for t, v in zip(obs_times, obs_values):
    ax_obs.plot([t, t], [v-noise_std, v+noise_std], 'b-', alpha=0.3, linewidth=1)

for t in obs_times:
    ax_obs.axvline(t, color='blue', linestyle='--', alpha=0.2, linewidth=1)

ax_obs.set_xlabel('Time (timesteps)', fontsize=10)
ax_obs.set_ylabel('Value', fontsize=10)
ax_obs.set_title('Observations: $\mathbf{y}$ (Only 9 points, $\sigma=0.05$, Very Sparse)', 
                 fontsize=11, fontweight='bold')
ax_obs.legend(loc='upper right', fontsize=9)
ax_obs.grid(True, alpha=0.3)
ax_obs.set_xlim(-5, 125)

# X(t)
ax_x = fig.add_subplot(gs[1, 0])
ax_x.plot(time_all, x_true_all[:, 0], color='#FF6B6B', linewidth=2.5, label='$x(t)$')
ax_x.set_xlabel('Time', fontsize=10)
ax_x.set_ylabel('$x$ value', fontsize=10)
ax_x.set_title('X-Component: $x(t)$ (First 120 steps)', fontsize=11, fontweight='bold', color='#FF6B6B')
ax_x.grid(True, alpha=0.3)
ax_x.legend(fontsize=9)

# Y(t)
ax_y = fig.add_subplot(gs[1, 1])
ax_y.plot(time_all, x_true_all[:, 1], color='#4ECDC4', linewidth=2.5, label='$y(t)$')
ax_y.set_xlabel('Time', fontsize=10)
ax_y.set_ylabel('$y$ value', fontsize=10)
ax_y.set_title('Y-Component: $y(t)$ (First 120 steps)', fontsize=11, fontweight='bold', color='#4ECDC4')
ax_y.grid(True, alpha=0.3)
ax_y.legend(fontsize=9)

# Z(t)
ax_z = fig.add_subplot(gs[1, 2])
ax_z.plot(time_all, x_true_all[:, 2], color='#95E1D3', linewidth=2.5, label='$z(t)$')
ax_z.set_xlabel('Time', fontsize=10)
ax_z.set_ylabel('$z$ value', fontsize=10)
ax_z.set_title('Z-Component: $z(t)$ (First 120 steps)', fontsize=11, fontweight='bold', color='#95E1D3')
ax_z.grid(True, alpha=0.3)
ax_z.legend(fontsize=9)

# Add text annotation explaining the problem

problem_text = (
    "THE INVERSE PROBLEM:\n"
)

fig.text(0.5, 0.02, problem_text, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4),
         style='italic', fontweight='bold')

plt.tight_layout(rect=[0, 0.06, 1, 1])


output_path = Path(__file__).parent / 'inverse_problem_visualization_lo.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved to: {output_path}")

plt.show()
