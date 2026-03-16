#!/usr/bin/env python

import argparse
import h5py

from dawgz import job, after, schedule
from pathlib import Path
from typing import *

from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PATH,
                        help='Root output directory (default: $SCRATCH/sda/lorenz or .)')
    args = parser.parse_args()

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / 'data').mkdir(parents=True, exist_ok=True)

    @job(cpus=1, ram='1GB', time='00:05:00')
    def simulate():
        chain = make_chain()

        x = chain.prior((1024,))
        x = chain.trajectory(x, length=1024, last=True)
        x = chain.trajectory(x, length=1024)
        x = chain.preprocess(x)
        x = x.transpose(0, 1)

        i = int(0.8 * len(x))
        j = int(0.9 * len(x))

        splits = {
            'train': x[:i],
            'valid': x[i:j],
            'test': x[j:],
        }
        for name, x in splits.items():
            with h5py.File(root / f'data/{name}.h5', mode='w') as f:
                f.create_dataset('x', data=x, dtype=np.float32)

    simulate()