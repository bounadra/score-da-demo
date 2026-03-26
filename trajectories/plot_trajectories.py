#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

data_file = "obs.h5"

# Load data
data_basename = Path(data_file).stem
with h5py.File(data_file, 'r') as f:
    keys = list(f.keys())
    
    if 'x' in f:
        # Full trajectories
        x = f['x'][:]  # Shape: (103, 1024, 3)
        data_type = "Trajectories"
    elif 'hi' in f or 'lo' in f:
        if 'hi' in f:
            x = f['hi'][:]  
            data_type = "Observations (hi)"
        else:
            x = f['lo'][:]
            data_type = "Observations (lo)"
    else:
        raise KeyError(f"Could not find 'x', 'hi', or 'lo' datasets in {data_file}. Available keys: {keys}")

if x.ndim == 2:
    x = np.expand_dims(x, axis=2)  
elif x.ndim == 3 and x.shape[2] == 1:
    x = np.squeeze(x, axis=2)
    x = np.expand_dims(x, axis=2) 

print(f"Loaded {x.shape[0]} {data_type} from {data_file}")
print(f"  Shape: {x.shape}")

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
traj = x[0]
if x.shape[2] >= 3:
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.7, linewidth=0.5)
    ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=50, label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=50, label='End')
elif x.shape[2] == 2:
    ax1.plot(traj[:, 0], traj[:, 1], np.zeros(len(traj)), 'b-', alpha=0.7, linewidth=0.5)
    ax1.scatter(traj[0, 0], traj[0, 1], 0, c='g', s=50, label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], 0, c='r', s=50, label='End')
else:  # x.shape[2] == 1
    t = np.arange(len(traj))
    ax1.plot(t, traj[:, 0], np.zeros(len(traj)), 'b-', alpha=0.7, linewidth=0.5)
    ax1.scatter(0, traj[0, 0], 0, c='g', s=50, label='Start')
    ax1.scatter(len(traj)-1, traj[-1, 0], 0, c='r', s=50, label='End')
ax1.set_xlabel('Dim 0')
ax1.set_ylabel('Dim 1')
ax1.set_zlabel('Dim 2')
ax1.set_title('3D Trajectory #0')
ax1.legend()


ax2 = fig.add_subplot(2, 3, 2, projection='3d')
colors = plt.cm.viridis(np.linspace(0, 1, 6))
for i in range(6):
    traj = x[i]
    if x.shape[2] >= 3:
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], alpha=0.6, linewidth=0.5)
    elif x.shape[2] == 2:
        ax2.plot(traj[:, 0], traj[:, 1], np.zeros(len(traj)), color=colors[i], alpha=0.6, linewidth=0.5)
    else:  # x.shape[2] == 1
        t = np.arange(len(traj))
        ax2.plot(t, traj[:, 0], np.zeros(len(traj)), color=colors[i], alpha=0.6, linewidth=0.5)
ax2.set_xlabel('Dim 0')
ax2.set_ylabel('Dim 1')
ax2.set_zlabel('Dim 2')
ax2.set_title('6 First Trajectories (3D)')

ax3 = fig.add_subplot(2, 3, 3)
for i in range(6):
    if x.shape[2] >= 2:
        ax3.plot(x[i, :, 0], x[i, :, 1], color=colors[i], alpha=0.6, linewidth=0.8, label=f'Traj {i}')
    else:
        t = np.arange(x.shape[1])
        ax3.plot(t, x[i, :, 0], color=colors[i], alpha=0.6, linewidth=0.8, label=f'Traj {i}')
ax3.set_xlabel('Time' if x.shape[2] == 1 else 'Dim 0')
ax3.set_ylabel('Dim 0' if x.shape[2] == 1 else 'Dim 1')
ax3.set_title('Time vs Dim 0 (6 trajectories)' if x.shape[2] == 1 else 'Dim 0-1 Projection (6 trajectories)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(2, 3, 4)
t = np.arange(x.shape[1])
ax4.plot(t, x[0, :, 0], 'r-', alpha=0.7, label='Dim 0')
if x.shape[2] >= 2:
    ax4.plot(t, x[0, :, 1], 'g-', alpha=0.7, label='Dim 1')
if x.shape[2] >= 3:
    ax4.plot(t, x[0, :, 2], 'b-', alpha=0.7, label='Dim 2')
ax4.set_xlabel('Time step')
ax4.set_ylabel('Value')
ax4.set_title('Time Series - Trajectory #0')
ax4.legend()
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(2, 3, 5)
for i in range(6):
    if x.shape[2] >= 3:
        ax5.plot(x[i, :, 0], x[i, :, 2], color=colors[i], alpha=0.6, linewidth=0.8)
        ax5.set_ylabel('Dim 2')
    else:
        ax5.plot(x[i, :, 0], np.arange(x.shape[1]), color=colors[i], alpha=0.6, linewidth=0.8)
        ax5.set_ylabel('Time')
ax5.set_xlabel('Dim 0')
ax5.set_title('Dim 0 vs Dim 2 (6 trajectories)' if x.shape[2] >= 3 else 'Dim 0 vs Time (6 trajectories)')
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
for i in range(6):
    if x.shape[2] >= 3:
        ax6.plot(x[i, :, 1], x[i, :, 2], color=colors[i], alpha=0.6, linewidth=0.8)
        ax6.set_xlabel('Dim 1')
        ax6.set_ylabel('Dim 2')
    elif x.shape[2] == 2:
        ax6.scatter(x[i, :, 0], x[i, :, 1], color=colors[i], alpha=0.6, s=10)
        ax6.set_xlabel('Dim 0')
        ax6.set_ylabel('Dim 1')
    else:  # x.shape[2] == 1
        ax6.scatter(i, np.mean(x[i, :, 0]), color=colors[i], s=100, alpha=0.6)
ax6.set_title('Dim 1-2 Projection (6 trajectories)' if x.shape[2] >= 3 else ('Dim 0-1 Scatter (6 trajectories)' if x.shape[2] == 2 else 'Mean values (1D data)'))
ax6.grid(alpha=0.3)

plt.tight_layout()
output_file = f'lorenz_trajectories_{data_basename}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")


# Print statistics
print("\nData statistics:")
for i in range(x.shape[2]):
    print(f"  Dim {i} - Min: {x[:, :, i].min():.3f}, Max: {x[:, :, i].max():.3f}, Mean: {x[:, :, i].mean():.3f}, Std: {x[:, :, i].std():.3f}")
