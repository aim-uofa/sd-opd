#!/bin/bash
set -eo pipefail

# Usage: HF_CHECKPOINT=/path/to/Qwen3-4B \
#          CUDA_VISIBLE_DEVICES=0,1 bash eval/eval_qwen3.sh qwen3-4b
# Models: qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-4b-instruct-2507,
#         qwen3-30b-a3b-instruct-2507. Extra arguments go to slime/train.py.
# Requires the Slime environment, including SGLang and Megatron-LM.
# Run download_data.sh to prepare data/eval/*.jsonl under the repository root.
# EVAL_CONFIG_PATH overrides the eval config; RESULT_DIR selects the output directory.
# TP_SIZE overrides GPUs per engine. NUM_GPUS defaults to the visible GPU count
# (8 when CUDA_VISIBLE_DEVICES is unset). Stop an old Ray head before rerunning.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
MODEL="${1:-qwen3-4b}"
if [ "$#" -gt 0 ]; then shift; fi

case "${MODEL,,}" in
   qwen3-1.7b)
      MODEL_NAME=Qwen3-1.7B
      TP_SIZE="${TP_SIZE:-1}"
      ;;
   qwen3-4b)
      MODEL_NAME=Qwen3-4B
      TP_SIZE="${TP_SIZE:-2}"
      ;;
   qwen3-8b)
      MODEL_NAME=Qwen3-8B
      TP_SIZE="${TP_SIZE:-2}"
      ;;
   qwen3-4b-instruct-2507)
      MODEL_NAME=Qwen3-4B-Instruct-2507
      TP_SIZE="${TP_SIZE:-2}"
      ;;
   qwen3-30b-a3b-instruct-2507)
      MODEL_NAME=Qwen3-30B-A3B-Instruct-2507
      TP_SIZE="${TP_SIZE:-8}"
      ;;
   *)
      echo "Unsupported model: ${MODEL}" >&2
      exit 1
      ;;
esac
if [ "${MODEL_NAME}" = Qwen3-30B-A3B-Instruct-2507 ]; then
   # Match slime/scripts/run-qwen3-30B-A3B.sh for the Instruct-2507 checkpoint.
   MODEL_ARGS_ROTARY_BASE=10000000 source "${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh"
else
   source "${SLIME_ROOT}/scripts/models/${MODEL_NAME,}.sh"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
   IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
   NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
else
   NUM_GPUS="${NUM_GPUS:-8}"
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ && "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
   (( NUM_GPUS % TP_SIZE != 0 )); then
   echo "NUM_GPUS must be a positive multiple of TP_SIZE." >&2
   exit 1
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/models/${MODEL_NAME}}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-${SCRIPT_DIR}/math_multi_task.yaml}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/${MODEL_NAME,,}}"
mkdir -p "${RESULT_DIR}"

export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
)

# These batch sizes satisfy Slime's shared train/eval interface; no training runs.
ROLLOUT_ARGS=(
   --num-rollout 0
   --debug-rollout-only
   --rollout-batch-size 8
   --n-samples-per-prompt 1
   --global-batch-size 8
   --micro-batch-size 1
   --rollout-seed 42
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-config "${EVAL_CONFIG_PATH}"
   --eval-max-context-len 40960
   --log-passrate
   --dump-details "${RESULT_DIR}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP_SIZE}"
   --sglang-enable-deterministic-inference
   --sglang-server-concurrency 128
   --sglang-context-length 40960
   --sglang-mem-fraction-static 0.7
)

# Follow slime/examples/eval_multi_task/multi_task.sh for Ray submission.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
keys = ("PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS")
print(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))
')

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "$@" 2>&1 | tee "${RESULT_DIR}/eval.log"
