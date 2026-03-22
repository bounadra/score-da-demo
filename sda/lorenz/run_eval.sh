#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --job-name=sda-lorenz-eval
#SBATCH --output=/Odyssey/private/c23bouna/UE_G/score-da-demo/logs/eval_%j.log
#SBATCH --error=/Odyssey/private/c23bouna/UE_G/score-da-demo/logs/eval_%j.err

set -euo pipefail

# Avoid potential NVML init issues with PyTorch
export PYTORCH_NO_NVML=1

# Keep HOME consistent on the cluster
export HOME=/Odyssey/private/c23bouna
unset SLURM_MEM_PER_CPU

# Project paths
PROJECT_DIR=/Odyssey/private/c23bouna/UE_G/score-da-demo

# Explicit paths for eval.py inputs/outputs
export SDA_DATA_PATH="${PROJECT_DIR}/data/data"
export SDA_RUNS_PATH="${PROJECT_DIR}/outputs/runs"
export SDA_RESULTS_PATH="${PROJECT_DIR}/outputs/results_eval"
export SDA_OBS_PATH="${PROJECT_DIR}/obs"
# Activate conda environment
source /Odyssey/private/c23bouna/miniforge3/etc/profile.d/conda.sh
conda activate scoreda

# Ensure log directory exists (SBATCH output path directory must exist)
mkdir -p "${PROJECT_DIR}/logs"

# Run evaluation (resume is handled by eval.py)
cd "${PROJECT_DIR}"
srun python sda/lorenz/eval.py
