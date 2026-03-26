# score-da-demo

Demonstration and evaluation of Score-Based Data Assimilation (SDA) vs classical 4D-Var on the Lorenz system.

This is an implementation of [Score-based Data Assimilation](https://arxiv.org/abs/2306.10574) by Rozet & Louppe (2023), with a comparative baseline using 4D-Variational methods.

## What's in here

**SDA approach**: Uses diffusion models to learn score functions over trajectory segments. At inference time, we generate trajectories conditioned on sparse noisy observations. The neat thing is the observation model is decoupled from training, so you can do zero-shot inference on different observation scenarios.

**4D-Var baseline**: Standard variational approach—minimize a cost function balancing model constraints and observation fit.

Both are evaluated on the Lorenz system with two scenarios:
- `lo`: sparse observations (every 8 steps, low noise σ=0.05)
- `hi`: dense observations (every step, higher noise σ=0.25)

## setup

```bash
cd sda
conda env create -f environment.yml
conda activate sda
pip install -e .
pip install git+https://github.com/google/jax-cfd
wandb login
```

Requires CUDA 11.7, Python 3.9+, and a Slurm cluster if you want to run the evaluation jobs.

## running things

**Generate data** (Lorenz trajectories):
```bash
cd sda/lorenz
python generate.py --output ./
```
Creates `data/{train,valid,test}.h5` (80/10/10 split).

**Train SDA** (score networks):
```bash
python train.py
```
Saves checkpoints to `runs/`. Logs to W&B under project `sda-lorenz`.

**Evaluate SDA**:
```bash
python eval.py
```
Runs inference on test set with observations, outputs `results_eval/stats_*.csv`.

**4D-Var baseline**:
```bash
cd ../../4D-var/lorenz
python eval.py
```

**On a cluster**:
```bash
sbatch run_eval_slurm.sh
sbatch ../../../4D-var/lorenz/submit_eval_slurm.sh
```

## structure

- `sda/` – main package with score networks, Markov chains, utilities
- `sda/lorenz/` – experiments on Lorenz: training and evaluation code
- `4D-var/lorenz/` – 4D-Var solver (baseline)
- `data/` – training/test trajectories (HDF5)
- `outputs/` – results and trained model checkpoints
- `trajectories/` – analysis scripts

## architecture

**Score networks**:
- `ScoreNet`: time-conditioned MLP with Fourier embeddings
- `ScoreUNet`: U-Net variant for spatial data
- `TimeEmbedding`: Fourier features for continuous time

**Training**: Denoising score matching on trajectory windows (default 5 steps). Losses tracked with W&B.

**Inference**: Reverse SDE sampling, conditioned on observations via Langevin dynamics.

## dependencies

Core: PyTorch (1.13.1), JAX (0.4.17), JAX-CFD, POT, Zuko

HPC/tracking: DAWGZ (job scheduling), W&B (experiment logging)

See `sda/environment.yml` for full list.

## env vars

- `SDA_RUNS_PATH`: where trained models go (default: `./runs/`)
- `SDA_DATA_PATH`: where data lives (default: `./data/`)
- `SDA_RESULTS_PATH`: output results dir (default: `./results_eval/`)
- `SDA_OBS_PATH`: observation data (default: `./obs/`)

## references

- Rozet & Louppe (2023): [Score-based Data Assimilation](https://arxiv.org/abs/2306.10574)
- Song et al.: Diffusion models and score-based generative modeling
- Evensen: Ensemble and variational data assimilation

## license

Based on implementation by [François Rozet](https://github.com/francois-rozet) and [Gilles Louppe](https://github.com/glouppe).
