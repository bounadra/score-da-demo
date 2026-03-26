#!/usr/bin/env python

import json
import os
import h5py
import numpy as np
import csv

from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *

# Set random seeds for reproducibility.
import random
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DATA_PATH = Path(os.environ.get('SDA_DATA_PATH', PATH / 'data'))
RUNS_PATH = Path(os.environ.get('SDA_RUNS_PATH', PATH / 'runs'))
RESULTS_PATH = Path(os.environ.get('SDA_RESULTS_PATH', PATH / 'results_eval_accelerated_64_step_C_diff_2'))
OBS_PATH = Path(os.environ.get('SDA_OBS_PATH', PATH / 'obs'))

RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_completed() -> Set[Tuple[int, str, str, int]]:
    """Load already-computed model evaluations (i, name, freq, C) from CSV files."""
    completed = set()
    
    for csv_path in [RESULTS_PATH / 'stats_lo.csv', RESULTS_PATH / 'stats_hi.csv']:
        if not csv_path.exists():
            continue
        
        freq = 'lo' if 'lo' in csv_path.name else 'hi'
        
        with open(csv_path, mode='r') as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                
                i_str, name, C_str = row[0], row[1], row[2]
                
                # Ground-truth rows are not keyed in the resume set.
                if C_str == '':
                    continue
                
                try:
                    i = int(i_str)
                    C = int(C_str)
                    completed.add((i, name, freq, C))
                except (ValueError, IndexError):
                    continue
    
    return completed


def discover_runs() -> List[Tuple[str, bool]]:
    runs: List[Tuple[str, bool]] = []

    if not RUNS_PATH.exists():
        return runs

    # Collect all runs first
    all_run_dirs = {}
    for run_dir in sorted(RUNS_PATH.iterdir()):
        if not run_dir.is_dir():
            continue

        state_file = run_dir / 'state.pth'
        config_file = run_dir / 'config.json'

        if not state_file.exists() or not config_file.exists():
            continue

        with open(config_file, mode='r') as f:
            config = json.load(f)

        # Local models have a "window" parameter in saved config.
        local = 'window' in config
        all_run_dirs[run_dir.name] = local

    # Custom order by run names
    custom_order = [
        'dummy-24qhkmqq_24qhkmqq',
        'dummy-aipkkdnl_aipkkdnl',
        'dummy-50hcas6u_50hcas6u',
        'dummy-celnvmn1_celnvmn1',
        'dummy-ycnhw3lr_ycnhw3lr',
        'dummy-uk32q4hm_uk32q4hm',
    ]
    
    for run_name in custom_order:
        if run_name in all_run_dirs:
            runs.append((run_name, all_run_dirs[run_name]))
    
    # Add any remaining runs not in custom order
    for run_name in sorted(all_run_dirs.keys()):
        if run_name not in custom_order:
            runs.append((run_name, all_run_dirs[run_name]))

    return runs


def observations():
    import random
    import os
    obs_file = OBS_PATH / 'obs.h5' if OBS_PATH.is_dir() else OBS_PATH
    if obs_file.exists():
        print(f"{obs_file} already exists, skipping creation.")
        return
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with h5py.File(DATA_PATH / 'test.h5', mode='r') as f:
        x = f['x'][:, :65]

    y_lo = np.random.normal(x[:, ::8, :1], 0.05)
    y_hi = np.random.normal(x[:, :, :1], 0.25)

    with h5py.File(obs_file, mode='w') as f:
        f.create_dataset('lo', data=y_lo)
        f.create_dataset('hi', data=y_hi)


def evaluation(i: int, name: str, local: bool, freq: str, completed: Set[Tuple[int, str, str, int]]):
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    corrections = (0, 1, 4, 8)
    # corrections = (2,)
    has_any_model = any((i, name, freq, C) in completed for C in corrections)

    # Skip fully completed evaluations early.
    if all((i, name, freq, C) in completed for C in corrections):
        print(f'{name} {freq} i={i:02d}: (fully skipped)', flush=True)
        return

    chain = make_chain()

    # Observation
    with h5py.File(OBS_PATH / 'obs.h5', mode='r') as f:
        y = torch.from_numpy(f[freq][i])

    A = lambda x: chain.preprocess(x)[..., :1]

    if freq == 'lo':  # low frequency & low noise
        sigma, step = 0.05, 8
    else:             # high frequency & high noise
        sigma, step = 0.25, 1

    # Ground truth particles are always needed for w1.
    # x = posterior(y, A=A, sigma=sigma, step=step)[:1024]
    # x_ = posterior(y, A=A, sigma=sigma, step=step)[:1024]

    _gt = posterior(y, A=A, sigma=sigma, step=step, particles=2048)
    x  = _gt[:1024]
    x_ = _gt[1024:]

    # Preserve historical layout: one GT row at the start of each run block.
    # If this block already has model rows, it is a resume and GT is not rewritten.
    if not has_any_model:

        log_px = log_prior(x).mean().item()
        log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
        w1 = emd(x, x_).item()

        with open(RESULTS_PATH / f'stats_{freq}.csv', mode='a') as f:
            f.write(f'{i},ground-truth,,{log_px},{log_py},{w1}\n')

        print('GT:', log_px, log_py, w1, flush=True)
    else:
        print('GT: (skipped, block already started)', flush=True)

    # Score
    score = load_score(RUNS_PATH / f'{name}/state.pth', local=local)
    sde = VPSDE(
        GaussianScore(
            y=y,
            A=lambda x: x[..., ::step, :1],
            std=sigma,
            sde=VPSDE(score, shape=()),
            gamma=3e-2,
        ),
        shape=(65, 3),
    ).cuda()

    for C in corrections:
        if (i, name, freq, C) in completed:
            print(f'{C:02d}: (skipped, already computed)', flush=True)
            continue
        
        
        # x = sde.sample((1024,), steps=32, corrections=C, tau=0.25).cpu()
        x = sde.sample((1024,), steps=64, corrections=C, tau=0.25).cpu()
        # x = sde.sample((1024,), steps=128, corrections=C, tau=0.25).cpu()
        # x = sde.sample((1024,), steps=256, corrections=C, tau=0.25).cpu()


        x = chain.postprocess(x)

        log_px = log_prior(x).mean().item()
        log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
        w1 = emd(x, x_).item()

        with open(RESULTS_PATH / f'stats_{freq}.csv', mode='a') as f:
            f.write(f'{i},{name},{C},{log_px},{log_py},{w1}\n')

        print(f'{C:02d}:', log_px, log_py, w1, flush=True)

if __name__ == '__main__':
    observations()

    runs = discover_runs()

    if not runs:
        raise FileNotFoundError(
            f'No run checkpoints found in {RUNS_PATH}. Expected folders with state.pth and config.json.'
        )

    completed = load_completed()
    print(f'Resuming: {len(completed)} evaluations already completed', flush=True)

    for name, local in runs:
        for freq in ['lo', 'hi']:
            for i in range(64):
                evaluation(i=i, name=name, local=local, freq=freq, completed=completed)
