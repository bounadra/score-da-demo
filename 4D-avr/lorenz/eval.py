#!/usr/bin/env python

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np
import torch


from solver import evaluate_reference, evaluate_weak_4dvar
from config import SCENARIOS


import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





PATH = Path(__file__).parent.resolve()
PROJECT_ROOT = PATH.parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "data" / "test.h5"
RESULTS_PATH = PATH / "outputs" / "results_seed_42"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_OBS_FILE = PROJECT_ROOT / "obs" / "obs.h5"


def ensure_observations(obs_file: Path, data_file: Path, seed: int = 42) -> Path:
    obs_file.parent.mkdir(parents=True, exist_ok=True)

    if obs_file.exists():
        return obs_file

    np.random.seed(seed)
    with h5py.File(data_file, mode="r") as f:
        x = f["x"][:, :65]

    with h5py.File(obs_file, mode="w") as f:
        for freq, cfg in SCENARIOS.items():
            y = np.random.normal(x[:, :: cfg.obs_step, :1], cfg.sigma_y)
            f.create_dataset(freq, data=y)

    return obs_file


def observations(obs_file: Path, data_file: Path, seed: int = 42) -> Path:
    return ensure_observations(obs_file=obs_file, data_file=data_file, seed=seed)


def open_writer(csv_path: Path):
    f = open(csv_path, mode="a", newline="")
    writer = csv.writer(f)
    return f, writer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=RESULTS_PATH)
    parser.add_argument("--n_obs", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--obs_file", type=Path, default=DEFAULT_OBS_FILE)
    parser.add_argument(
        "--background_std",
        type=float,
        default=None,
        help="Optional override for weak-4DVar background standard deviation.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional override for weak-4DVar LBFGS max iterations.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "1")))
    torch.set_num_threads(max(1, cpu_threads))

    torch.manual_seed(args.seed)

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    obs_file = observations(obs_file=args.obs_file, data_file=args.data_file, seed=args.seed)

    for freq in SCENARIOS:
        cfg = SCENARIOS[freq]
        background_std = cfg.background_std if args.background_std is None else args.background_std
        maxiter = cfg.maxiter if args.maxiter is None else args.maxiter

        csv_path = root / f"stats_{freq}.csv"
        fout, writer = open_writer(csv_path)

        try:
            with h5py.File(obs_file, mode="r") as f:
                y_all = f[freq][: args.n_obs]
                
                
                for i in range(len(y_all)):
                    y = torch.from_numpy(y_all[i]).float().to(device)

                    x_ref, gt_stats = evaluate_reference(
                        y=y,
                        freq=freq,
                        n_samples=args.n_samples,
                        device=device,
                    )
                    writer.writerow([i, "ground-truth", "", gt_stats["log_px"], gt_stats["log_py"], gt_stats["w1"]])

                    w4d_stats = evaluate_weak_4dvar(
                        y=y,
                        freq=freq,
                        x_ref=x_ref,
                        n_samples=args.n_samples,
                        background_std=background_std,
                        maxiter=maxiter,
                        device=device,
                    )
                    writer.writerow([i, "weak-4dvar", maxiter, w4d_stats["log_px"], w4d_stats["log_py"], w4d_stats["w1"]])

                    fout.flush()
                    print(
                        f"[{freq}] sample {i + 1}/{len(y_all)} done (background_std={background_std}, maxiter={maxiter})",
                        flush=True,
                    )

        finally:
            fout.close()

        print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()