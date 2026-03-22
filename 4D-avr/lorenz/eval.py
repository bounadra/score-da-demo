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



PATH = Path(__file__).parent.resolve()
PROJECT_ROOT = PATH.parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "data" / "test.h5"
RESULTS_PATH = PATH / "outputs" / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_CORRECTIONS = (0, 1, 2, 4, 8, 16)


def ensure_observations(root: Path, data_file: Path, seed: int = 0) -> Path:
    results_dir = root
    results_dir.mkdir(parents=True, exist_ok=True)
    obs_file = results_dir / "obs.h5"

    if obs_file.exists():
        return obs_file

    rng = np.random.default_rng(seed)
    with h5py.File(data_file, mode="r") as f:
        x = f["x"][:, :65]

    with h5py.File(obs_file, mode="w") as f:
        for freq, cfg in SCENARIOS.items():
            y = rng.normal(x[:, :: cfg.obs_step, :1], cfg.sigma_y)
            f.create_dataset(freq, data=y)

    return obs_file


def observations(root: Path, data_file: Path, seed: int = 0) -> Path:
    return ensure_observations(root=root, data_file=data_file, seed=seed)


def open_writer(csv_path: Path):
    f = open(csv_path, mode="a", newline="")
    writer = csv.writer(f)
    return f, writer


def parse_corrections(raw: str):
    values = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not values:
        raise ValueError("At least one correction value is required.")
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PATH)
    parser.add_argument("--n_obs", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--weak_iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--corrections",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CORRECTIONS),
        help="Comma-separated correction/iteration values to evaluate (e.g. 0,1,2,4,8,16).",
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
    corrections = parse_corrections(args.corrections)

    torch.manual_seed(args.seed)

    root = args.root
    obs_file = observations(root=root, data_file=args.data_file, seed=args.seed)

    for freq in SCENARIOS:
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

                    for corr in corrections:
                        w4d_stats = evaluate_weak_4dvar(
                            y=y,
                            freq=freq,
                            x_ref=x_ref,
                            n_samples=args.n_samples,
                            iterations=corr,
                            device=device,
                        )
                        writer.writerow([i, "weak-4dvar", corr, w4d_stats["log_px"], w4d_stats["log_py"], w4d_stats["w1"]])

                    fout.flush()
                    print(f"[{freq}] sample {i + 1}/{len(y_all)} done", flush=True)

        finally:
            fout.close()

        print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()