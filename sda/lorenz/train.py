#!/usr/bin/env python

import os
import wandb

from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


RUNS_PATH = Path(os.environ.get('SDA_RUNS_PATH', PATH / 'runs'))
DATA_PATH = Path(os.environ.get('SDA_DATA_PATH', PATH / 'data'))
RUNS_PATH.mkdir(parents=True, exist_ok=True)


GLOBAL_CONFIG = {
    # Architecture
    'embedding': 32,
    'hidden_channels': (64,),
    'hidden_blocks': (3,),
    'activation': 'SiLU',
    # Training
    'epochs': 4096,
    'batch_size': 64,
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}

LOCAL_CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 32,
    'width': 256,
    'depth': 5,
    'activation': 'SiLU',
    # Training
    'epochs': 4096,
    'batch_size': 64,
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}


def train_global(i: int):
    run = wandb.init(project='sda-lorenz', group='global', config=GLOBAL_CONFIG)
    runpath = RUNS_PATH / f'{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(GLOBAL_CONFIG, runpath)

    # Network
    score = make_global_score(**GLOBAL_CONFIG)
    sde = VPSDE(score, shape=(32, 3)).cuda()

    # Data
    trainset = TrajectoryDataset(DATA_PATH / 'train.h5', window=32)
    validset = TrajectoryDataset(DATA_PATH / 'valid.h5', window=32)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **GLOBAL_CONFIG,
        device='cuda',
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation
    chain = make_chain()

    x = sde.sample((1024,), steps=64).cpu()
    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


def train_local(i: int):
    run = wandb.init(project='sda-lorenz', group='local', config=LOCAL_CONFIG)
    runpath = RUNS_PATH / f'{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(LOCAL_CONFIG, runpath)

    # Network
    window = LOCAL_CONFIG['window']
    score = make_local_score(**LOCAL_CONFIG)
    sde = VPSDE(score.kernel, shape=(window * 3,)).cuda()

    # Data
    trainset = TrajectoryDataset(DATA_PATH / 'train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(DATA_PATH / 'valid.h5', window=window, flatten=True)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **LOCAL_CONFIG,
        device='cuda',
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation
    chain = make_chain()

    x = sde.sample((4096,), steps=64).cpu()
    x = x.unflatten(-1, (-1, 3))
    x = chain.postprocess(x)

    log_p = chain.log_prob(x[:, :-1], x[:, 1:]).mean()

    run.log({'log_p': log_p})
    run.finish()


if __name__ == '__main__':
    os.environ.setdefault('WANDB_SILENT', 'true')

    for i in range(3):
        train_global(i)

    for i in range(3):
        train_local(i)