#!/usr/bin/env bash
set -euo pipefail

# Evaluate the original MiMo teacher using its native non-thinking template.
# HF_CHECKPOINT, EVAL_CONFIG_PATH, RESULT_DIR and TP_SIZE override defaults.
# Distilled Qwen3 checkpoints continue to use eval_checkpoint.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_ROOT}/models}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${WEIGHTS_DIR}/MiMo-7B-RL-0530}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-${SCRIPT_DIR}/math_multi_task.yaml}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/mimo-7b-rl-0530-nonthinking}"
TP_SIZE="${TP_SIZE:-2}"

source "${SLIME_ROOT}/scripts/models/mimo-7B-rl.sh"
# The 0530 checkpoint has a 64K model capacity; use the existing 40K eval budget.
MODEL_ARGS+=(--max-position-embeddings 65536)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
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
for required_path in "${HF_CHECKPOINT}/config.json" "${EVAL_CONFIG_PATH}"; do
   if [[ ! -f "${required_path}" ]]; then
      echo "Missing required file: ${required_path}" >&2
      exit 1
   fi
done
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

EVAL_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --num-rollout 0
   --debug-rollout-only
   --rollout-batch-size 8
   --n-samples-per-prompt 1
   --global-batch-size 8
   --micro-batch-size 1
   --rollout-seed 42
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking": false}'
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

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265
trap 'ray stop --force || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
keys = ("PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS")
print(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))
')
ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" \
   "$@" 2>&1 | tee "${RESULT_DIR}/eval.log"
