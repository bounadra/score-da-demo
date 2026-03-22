#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=240G
#SBATCH --job-name=4dvar-lorenz-eval
#SBATCH --output=/Odyssey/private/c23bouna/UE_G/score-da-demo/logs/4dvar_eval_%j.log
#SBATCH --error=/Odyssey/private/c23bouna/UE_G/score-da-demo/logs/4dvar_eval_%j.err

set -euo pipefail

# Avoid potential NVML init issues with PyTorch
export PYTORCH_NO_NVML=1

# Keep HOME consistent on the cluster
export HOME=/Odyssey/private/c23bouna
unset SLURM_MEM_PER_CPU

# Project paths
PROJECT_DIR=/Odyssey/private/c23bouna/UE_G/score-da-demo
RESULTS_4DVAR_DIR="${PROJECT_DIR}/4D-avr/lorenz/outputs/results_seed_42"
OBS_SOURCE_FILE="${PROJECT_DIR}/obs/obs.h5"

# Ensure local source tree is importable in batch jobs.
export PYTHONPATH="${PROJECT_DIR}/sda:${PROJECT_DIR}:${PYTHONPATH:-}"

# Activate conda environment
source /Odyssey/private/c23bouna/miniforge3/etc/profile.d/conda.sh
conda activate scoreda

# Ensure directories exist
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RESULTS_4DVAR_DIR}"

# Optional clean start for CSVs (uncomment if needed)
# rm -f "${RESULTS_4DVAR_DIR}/stats_lo.csv" "${RESULTS_4DVAR_DIR}/stats_hi.csv"

CPU_TASKS="${SLURM_CPUS_PER_GPU:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${PROJECT_DIR}"
srun --cpus-per-task="${CPU_TASKS}" python 4D-avr/lorenz/eval.py \
  --root "${RESULTS_4DVAR_DIR}" \
  --obs_file "${OBS_SOURCE_FILE}" \
  --n_obs 64 \
  --n_samples 1024 \
  --device cuda \
  --seed 0
