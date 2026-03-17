#!/usr/bin/env python

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import os


from solver import evaluate_reference, evaluate_weak_4dvar



PATH = Path(__file__).parent.resolve()
RESULTS_PATH = PATH / "outputs" / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

DEFAULT_RUNS: List[Tuple[str, bool]] = [
    ("polar-capybara-13_y1g6w4jm", True),
    ("snowy-leaf-29_711r6as1", True),
    ("ruby-serenity-42_nbhxlnf9", True),
    ("light-moon-51_09a36gw8", True),
    ("lilac-bush-61_7f0sioiw", False),
]


def ensure_observations(root: Path, seed: int = 0) -> Path:
    results_dir = root
    results_dir.mkdir(parents=True, exist_ok=True)
    obs_file = results_dir / "obs.h5"

    if obs_file.exists():
        return obs_file

    rng = np.random.default_rng(seed)
    with h5py.File(root.parent / "data/test.h5", mode="r") as f:
        x = f["x"][:, :65]

    y_lo = rng.normal(x[:, ::8, :1], 0.05)
    y_hi = rng.normal(x[:, :, :1], 0.25)

    with h5py.File(obs_file, mode="w") as f:
        f.create_dataset("lo", data=y_lo)
        f.create_dataset("hi", data=y_hi)

    return obs_file


def open_writer(csv_path: Path):
    file_exists = csv_path.exists()
    f = open(csv_path, mode="a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["i", "method", "corrections", "log_px", "log_py", "w1"])
    return f, writer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PATH)
    parser.add_argument("--n_obs", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--weak_iterations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_score", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = args.root
    obs_file = ensure_observations(root=root, seed=args.seed)

    for freq in ("lo", "hi"):
        csv_path = RESULTS_PATH / f"stats_{freq}.csv"
        fout, writer = open_writer(csv_path)

        try:
            with h5py.File(obs_file, mode="r") as f:
                y_all = f[freq][: args.n_obs]
                
                
                for i in range(len(y_all)):
                    y = torch.from_numpy(y_all[i]).float()

                    x_ref, gt_stats = evaluate_reference(
                        y=y,
                        freq=freq,
                        n_samples=args.n_samples,
                    )
                    writer.writerow([i, "ground-truth", "", gt_stats["log_px"], gt_stats["log_py"], gt_stats["w1"]])

                    w4d_stats = evaluate_weak_4dvar(
                        y=y,
                        freq=freq,
                        x_ref=x_ref,
                        n_samples=args.n_samples,
                        iterations=args.weak_iterations,
                    )
                    writer.writerow([i, "weak-4dvar", "", w4d_stats["log_px"], w4d_stats["log_py"], w4d_stats["w1"]])

                    fout.flush()
                    print(f"[{freq}] sample {i + 1}/{len(y_all)} done", flush=True)

        finally:
            fout.close()

        print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()