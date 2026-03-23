#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import h5py
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


import random
import numpy as np
import torch



ROOT_PATH = Path(__file__).resolve().parents[2]  # or adjust as needed

# Nom du dossier de résultats rendu dynamique
RESULT_DIR = 'results_dummy50'

print(f'Current working directory: {ROOT_PATH}' )

print("Looking for test.h5 at:", ROOT_PATH / 'data/data/test.h5')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

chain = make_chain()

colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))

rc = {
    'axes.axisbelow': True,
    'axes.linewidth': 1.0,
    'axes.prop_cycle': plt.cycler('color', colors),
    'figure.dpi': 150,
    'figure.figsize': (6.4, 4.8),
    'font.size': 12.0,
    'legend.fontsize': 'xx-small',
    'lines.linewidth': 1.0,
    'lines.markersize': 2.5,
    'savefig.transparent': True,
    'xtick.labelsize': 'x-small',
    'xtick.major.width': 1.0,
    'ytick.labelsize': 'x-small',
    'ytick.major.width': 1.0,
    'font.family': ['serif'],
    'text.usetex': True,
}

plt.rcParams.update(rc)

### Statistics

print("Opening file at:", ROOT_PATH / 'data/data/test.h5')

with h5py.File(ROOT_PATH / 'data/data/test.h5', mode='r') as f:
    x_star = torch.from_numpy(f['x'][1, :257])
    y_star = torch.normal(x_star[::8, :1], 0.05)

A = lambda x: chain.preprocess(x)[..., :1]
sigma = 0.05
step = 8

x_bpf = posterior(y_star, A, sigma, step, particles=2**16)[:1024]

APPROXS = {
    # 2: 'dummy-24qhkmqq_24qhkmqq',
    2: 'dummy-50hcas6u_50hcas6u'
    # 2: 'dummy-ycnhw3lr_ycnhw3lr'
    }

fig, axs = plt.subplot_mosaic([
    ['A', 'A', 'B', 'B', 'C', 'C', 0, 0, 0],
    ['A', 'A', 'B', 'B', 'C', 'C', 0, 0, 0],
    ['A', 'A', 'B', 'B', 'C', 'C', 1, 1, 1],
    ['D', 'D', 'E', 'E', 'F', 'F', 1, 1, 1],
    ['D', 'D', 'E', 'E', 'F', 'F', 2, 2, 2],
    ['D', 'D', 'E', 'E', 'F', 'F', 2, 2, 2],
], figsize=(6.4, 3.2))

axs['B'].sharey(axs['A'])
axs['C'].sharey(axs['A'])
axs['E'].sharey(axs['D'])
axs['F'].sharey(axs['D'])

axs['A'].sharex(axs['D'])
axs['B'].sharex(axs['E'])
axs['C'].sharex(axs['F'])

axs[1].sharex(axs[0])
axs[2].sharex(axs[0])
    
# Statistics
coordinates = zip(
    'ABCDEF',
    itertools.product(('lo', 'hi'), ('log_px', 'log_py', 'w1'))
)

for key, (freq, column) in coordinates:
    ax = axs[key]    
    df = pd.read_csv(ROOT_PATH / f'outputs/{RESULT_DIR}/stats_{freq}.csv', header=None, names=['approx', 'corrections', 'log_px', 'log_py', 'w1'])
    df = df.groupby(['approx', 'corrections'], dropna=False).median().reset_index()

    ax.axvline(df[column][df['approx'] == 'ground-truth'].to_numpy(), ls='--', color='r', label='BPF')

    for k, approx in APPROXS.items():
        df_ = df[df['approx'] == approx]

        if k > 4:
            df_ = df_[df_['corrections'] > 0]

        ax.plot(df_[column], df_['corrections'], '-o', label=f'$k$={k}')

        if column == 'log_px':
            ax.set_xlabel('$\log p(x_{2:L} \mid x_1)$')
            ax.set_xlim(-299, 149)
        elif column == 'log_py':
            ax.set_xlabel('$\log p(y \mid x_{1:L})$')
            ax.set_xlim(-14, 19)
        elif column == 'w1':
            ax.set_xlabel('$W_1$')
            ax.set_xlim(-5, 59)

    if key == 'A':
        ax.text(-0.2, 0.025, r'\textbf{\textsf{A}}', transform=ax.transAxes, ha='center')
    elif key == 'D':
        ax.text(-0.2, 0.025, r'\textbf{\textsf{B}}', transform=ax.transAxes, ha='center')
    elif key == 'E':
        ax.legend()

    ax.set_ylabel('Corrections')
    ax.set_yscale('symlog', base=2, linthresh=1, linscale=0.5)
    ax.set_yticks([0, 1, 2, 4, 8, 16])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.tick_params(direction='in')
    ax.label_outer()
    ax.grid()

# Example
for i in range(3):
    ax = axs[i]

    ax.plot(
        range(1, 32 * 6 + 2),
        x_bpf[:64, 0:32 * 6 + 1, i].T,
        color='red',
        lw=0.5,
        alpha=0.05,
    )

    # for j, x in enumerate(xs):
    #     ax.plot(
    #         range(32 * j + 1, 32 * (j+1) + 2),
    #         x[:, 32 * j:32 * (j+1) + 1, i].T,
    #         color=f'C{j}',
    #         lw=0.5,
    #         alpha=0.1,
    #     )

    if i == 0:
        ax.set_ylabel('$a$')
        ax.set_yticks([-10, 0, 10])
    elif i == 1:
        ax.set_ylabel('$b$')
        ax.set_yticks([-15, 0, 15])
    elif i == 2:
        ax.set_ylabel('$c$')
        ax.set_yticks([10, 25, 40])
        ax.text(0.025, -0.375, r'\textbf{\textsf{C}}', transform=ax.transAxes)

    ax.set_xlabel('$i$')
    ax.set_xticks(range(1, 7 * 32 + 1, 32))

    ax.yaxis.set_label_position('right')
    ax.yaxis.label.set(rotation='horizontal', ha='left', va='center')
    ax.yaxis.tick_right()

    ax.tick_params(direction='in')
    ax.label_outer()
    ax.grid()

fig.tight_layout(pad=0.5)
fig.align_labels()
fig.savefig(f'statistics_{RESULT_DIR}.pdf', pad_inches=0.025, bbox_inches='tight')