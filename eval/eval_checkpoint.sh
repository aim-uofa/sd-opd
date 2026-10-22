#!/bin/bash
set -eo pipefail

# Evaluate a Megatron torch_dist checkpoint using Slime's HF conversion tool.
# BASE_MODEL must contain the matching original HF config and tokenizer.
# Usage: BASE_MODEL=/path/to/Qwen3-4B CHECKPOINT_PATH=/path/to/checkpoints \
#          CHECKPOINT_STEP=119 CUDA_VISIBLE_DEVICES=0,1 \
#          bash eval/eval_checkpoint.sh qwen3-4b
# Accepts the same model names, environment overrides and Slime arguments as
# eval_qwen3.sh. HF_CHECKPOINT_PATH selects the export directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
MODEL="${1:-qwen3-4b}"
if [ "$#" -gt 0 ]; then shift; fi

: "${BASE_MODEL:?Set BASE_MODEL to the matching original HF model directory.}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the Megatron checkpoint root.}"
if ! [[ "${CHECKPOINT_STEP:-}" =~ ^[1-9][0-9]*$ ]]; then
   echo "CHECKPOINT_STEP must be a positive integer." >&2
   exit 1
fi
printf -v CHECKPOINT_ITERATION 'iter_%07d' "${CHECKPOINT_STEP}"
TORCH_DIST_PATH="${CHECKPOINT_PATH}/${CHECKPOINT_ITERATION}"
if [ ! -d "${TORCH_DIST_PATH}" ] || [ ! -f "${BASE_MODEL}/config.json" ]; then
   echo "Check TORCH_DIST_PATH (${TORCH_DIST_PATH}) and BASE_MODEL (${BASE_MODEL})." >&2
   exit 1
fi

export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
HF_CHECKPOINT_PATH="${HF_CHECKPOINT_PATH:-${CHECKPOINT_PATH}/hf/${CHECKPOINT_ITERATION}}"
# A completed export is reusable. Slime refuses to overwrite an incomplete or
# existing directory; remove it or choose a new HF_CHECKPOINT_PATH to re-export.
if [ ! -f "${HF_CHECKPOINT_PATH}/config.json" ] ||
   ! compgen -G "${HF_CHECKPOINT_PATH}/*.safetensors" >/dev/null; then
   python3 "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py" \
      --input-dir "${TORCH_DIST_PATH}" \
      --output-dir "${HF_CHECKPOINT_PATH}" \
      --origin-hf-dir "${BASE_MODEL}" \
      --vocab-size 151936
fi

export HF_CHECKPOINT="${HF_CHECKPOINT_PATH}"
RUN_NAME="$(basename -- "${CHECKPOINT_PATH}")"
export RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${RUN_NAME}/${CHECKPOINT_ITERATION}}"
exec bash "${SCRIPT_DIR}/eval_qwen3.sh" "${MODEL}" "$@"
