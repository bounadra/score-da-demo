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
RESULTS_4DVAR_DIR="${PROJECT_DIR}/4D-avr/lorenz/outputs/results_seed_42_submit_v2"
OBS_SOURCE_FILE="${PROJECT_DIR}/obs/obs.h5"

# Ensure local source tree is importable in batch jobs.
export PYTHONPATH="${PROJECT_DIR}/sda:${PROJECT_DIR}:${PYTHONPATH:-}"

# Activate conda environment
source /Odyssey/private/c23bouna/miniforge3/etc/profile.d/conda.sh
conda activate scoreda

# Ensure directories exist
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RESULTS_4DVAR_DIR}"

CPU_TASKS="${SLURM_CPUS_PER_GPU:-${SLURM_CPUS_PER_TASK:-1}}"

JOB_ID="${SLURM_JOB_ID:-manual}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_4DVAR_DIR}/run_meta"

cp "${PROJECT_DIR}/4D-avr/lorenz/eval.py" \
  "${RESULTS_4DVAR_DIR}/run_meta/eval_${JOB_ID}_${STAMP}.py"

cp "${PROJECT_DIR}/4D-avr/lorenz/solver.py" \
  "${RESULTS_4DVAR_DIR}/run_meta/solver_${JOB_ID}_${STAMP}.py"

git -C "${PROJECT_DIR}" rev-parse HEAD \
  > "${RESULTS_4DVAR_DIR}/run_meta/git_commit_${JOB_ID}_${STAMP}.txt"

cd "${PROJECT_DIR}"
srun --cpus-per-task="${CPU_TASKS}" python 4D-avr/lorenz/eval.py \
  --root "${RESULTS_4DVAR_DIR}" \
  --obs_file "${OBS_SOURCE_FILE}" \
  --n_obs 64 \
  --n_samples 1024 \
  --device cuda \
  --seed 42
