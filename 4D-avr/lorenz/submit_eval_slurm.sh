#!/bin/bash
set -euo pipefail

PROJECT_DIR=/Odyssey/private/c23bouna/UE_G/score-da-demo
RUN_SCRIPT="${PROJECT_DIR}/4D-avr/lorenz/run_eval_slurm.sh"
ALLOW_SM90="${ALLOW_SM90:-0}"

if [[ "${1:-}" == "--allow-sm90" ]]; then
  ALLOW_SM90=1
fi

choose_gpu() {
  local gpu
  # Current scoreda PyTorch build supports up to sm_86, so prefer A100 (sm_80).
  for gpu in a100; do
    if sinfo -p Odyssey -h -o '%t %G' | awk -v pat="gpu:${gpu}" '$1 == "idle" && index($0, pat) {found=1} END{exit !found}'; then
      echo "${gpu}"
      return 0
    fi
  done

  if [[ "${ALLOW_SM90}" == "1" ]]; then
    for gpu in h100 h200; do
      if sinfo -p Odyssey -h -o '%t %G' | awk -v pat="gpu:${gpu}" '$1 == "idle" && index($0, pat) {found=1} END{exit !found}'; then
        echo "${gpu}"
        return 0
      fi
    done
  fi

  # Queue on compatible architecture by default.
  echo "a100"
}

GPU_TYPE="$(choose_gpu)"

submit_with_profile() {
  local exclusive_flag="$1"
  local cpus="$2"
  local mem="$3"

  if [[ -n "${exclusive_flag}" ]]; then
    sbatch \
      --partition=Odyssey \
      --gres="gpu:${GPU_TYPE}:1" \
      --cpus-per-gpu="${cpus}" \
      --mem="${mem}" \
      "${exclusive_flag}" \
      "${RUN_SCRIPT}"
  else
    sbatch \
      --partition=Odyssey \
      --gres="gpu:${GPU_TYPE}:1" \
      --cpus-per-gpu="${cpus}" \
      --mem="${mem}" \
      "${RUN_SCRIPT}"
  fi
}

echo "Submitting 4DVar eval on Odyssey with gpu:${GPU_TYPE}:1"
echo "Trying profiles: exclusive 32c/240G -> shared 24c/180G -> shared 16c/120G"

if JOB_OUT="$(submit_with_profile --exclusive 32 240G 2>&1)"; then
  echo "Submitted with profile: exclusive 32c/240G"
  echo "${JOB_OUT}"
  exit 0
fi

if JOB_OUT="$(submit_with_profile "" 24 180G 2>&1)"; then
  echo "Submitted with profile: shared 24c/180G"
  echo "${JOB_OUT}"
  exit 0
fi

if JOB_OUT="$(submit_with_profile "" 16 120G 2>&1)"; then
  echo "Submitted with profile: shared 16c/120G"
  echo "${JOB_OUT}"
  exit 0
fi

echo "All submission profiles failed. Last sbatch output:"
echo "${JOB_OUT}"
exit 1
