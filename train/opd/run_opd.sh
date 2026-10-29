#!/usr/bin/env bash
# Shared Slime OPD launcher; select a setting via train/opd/*.sh.
set -euo pipefail

TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAIN_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${TRAIN_DIR}/config.sh"

SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/Megatron-LM}"
export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
OPD_TYPE="${OPD_TYPE:-sglang}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_ROOT}/models}"
STUDENT_MODEL="${STUDENT_MODEL:-${WEIGHTS_DIR}/${STUDENT_NAME}}"
TEACHER_MODEL="${TEACHER_MODEL:-${WEIGHTS_DIR}/${TEACHER_NAME}}"
STUDENT_MCORE="${STUDENT_MCORE:-${STUDENT_MODEL}}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/dapo-math-17k/dapo-math-17k.jsonl}"
RUN_NAME="${TEACHER_NAME,,}-to-${STUDENT_NAME,,}-opd-nonthinking-seed-${TRAIN_SEED}"
if [[ "${OPD_TYPE}" == mimo ]]; then
   RUN_NAME="${TEACHER_NAME,,}-to-${STUDENT_NAME,,}-opd-shared-vocab-v3-seed-${TRAIN_SEED}"
   if [[ "${TP_SIZE}" != 1 ]]; then
      echo "MiMo shared-vocabulary OPD requires actor TP_SIZE=1." >&2
      exit 1
   fi
   for arg in "$@"; do
      case "${arg}" in
         --load|--load=*|--save|--save=*|--hf-checkpoint|--hf-checkpoint=*|--ref-load|--ref-load=*)
            echo "Set SAVE_DIR, STUDENT_MODEL or STUDENT_MCORE instead of ${arg}." >&2
            exit 1
            ;;
      esac
   done
fi
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/${RUN_NAME}}"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-${SAVE_DIR}/teacher.log}"
source "${SLIME_ROOT}/scripts/models/${STUDENT_NAME,}.sh"

# Keep the pilot's GPU counts; use Slime's TP teacher on the final GPU(s).
# CUDA_VISIBLE_DEVICES specifies physical GPUs, in actor/rollout/teacher order.
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_GPUS}"
case "${#GPU_IDS[@]}" in
   4) ACTOR_NUM_GPUS=2; ROLLOUT_NUM_GPUS=1; TEACHER_NUM_GPUS=1 ;;
   8) ACTOR_NUM_GPUS=2; ROLLOUT_NUM_GPUS=4; TEACHER_NUM_GPUS=2 ;;
   *) echo "CUDA_VISIBLE_DEVICES must select 4 or 8 GPUs." >&2; exit 1 ;;
esac
if [[ "${VISIBLE_GPUS}" == ,* || "${VISIBLE_GPUS}" == *, ||
      "${VISIBLE_GPUS}" == *,,* || "${VISIBLE_GPUS}" == *[[:space:]]* ]]; then
   echo "CUDA_VISIBLE_DEVICES contains an empty GPU ID or whitespace." >&2
   exit 1
fi
for ((i = 0; i < ${#GPU_IDS[@]}; i++)); do
   for ((j = 0; j < i; j++)); do
      if [[ "${GPU_IDS[i]}" == "${GPU_IDS[j]}" ]]; then
         echo "CUDA_VISIBLE_DEVICES contains duplicate GPU IDs." >&2
         exit 1
      fi
   done
done
TRAIN_GPU_COUNT=$((ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS))
TRAIN_GPUS="$(IFS=,; echo "${GPU_IDS[*]:0:TRAIN_GPU_COUNT}")"
TEACHER_GPUS="$(IFS=,; echo "${GPU_IDS[*]:TRAIN_GPU_COUNT}")"
CHAT_KWARGS='{"enable_thinking": false}'
REWARD_MODULE=slime.rollout.on_policy_distillation
REWARD_ENDPOINT=generate
MIMO_ARGS=()
if [[ "${OPD_TYPE}" == mimo ]]; then
   # MiMo is a frozen Transformers scorer on one GPU, including in 8-GPU runs.
   TEACHER_NUM_GPUS=1
   TEACHER_GPUS="${GPU_IDS[TRAIN_GPU_COUNT]}"
   CHAT_KWARGS="$(python3 -m slime.utils.shared_vocab)"
   REWARD_MODULE=slime.rollout.mimo_opd
   REWARD_ENDPOINT=score
   MIMO_ARGS=(
      --opd-shared-vocab-size 151665
      --custom-generate-function-path slime.rollout.shared_vocab.generate
      --sglang-enable-custom-logit-processor
      --sglang-context-length "${ROLLOUT_MAX_CONTEXT_LEN}"
      --rollout-stop-token-ids 151645 151643
   )
fi

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL}"
   --ref-load "${STUDENT_MCORE}"
   --load "${SAVE_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)
ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --apply-chat-template
   --apply-chat-template-kwargs "${CHAT_KWARGS}"
   --rollout-seed "${TRAIN_SEED}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --rollout-top-p "${ROLLOUT_TOP_P}"
   --rollout-top-k "${ROLLOUT_TOP_K}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
RM_ARGS=(
   --custom-rm-path "${REWARD_MODULE}.reward_func"
   --custom-reward-post-process-path "${REWARD_MODULE}.post_process_rewards"
   --rm-url "http://127.0.0.1:13141/${REWARD_ENDPOINT}"
)
OPD_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type "${OPD_TYPE}"
   --opd-kl-coef "${OPD_KL_COEF}"
   --eps-clip "${EPS_CLIP}"
   --entropy-coef "${ENTROPY_COEF}"
   --calculate-per-token-loss
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --seed "${TRAIN_SEED}"
   --lr "${LR}"
   --lr-decay-style "${LR_DECAY_STYLE}"
   --weight-decay "${WEIGHT_DECAY}"
   --adam-beta1 "${ADAM_BETA1}"
   --adam-beta2 "${ADAM_BETA2}"
   --clip-grad "${CLIP_GRAD}"
)
PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   # Preserve Qwen3's 151936-token vocabulary at both TP=1 and TP=2.
   --make-vocab-size-divisible-by 64
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}"
)
if (( TP_SIZE > 1 )); then
   PERF_ARGS+=(--sequence-parallel)
fi
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tb-project-name "${TB_PROJECT_NAME:-opd}"
   --tb-experiment-name "${TB_EXPERIMENT_NAME:-${RUN_NAME}}"
)
TRAIN_CMD=(
   python3 "${SLIME_ROOT}/train.py"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --num-gpus-per-node "${TRAIN_GPU_COUNT}"
   "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}" "${OPD_ARGS[@]}" "${OPTIMIZER_ARGS[@]}"
   "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${MIMO_ARGS[@]}" "$@"
)
TEACHER_CMD=(
   python3 -m sglang.launch_server
   --model-path "${TEACHER_MODEL}"
   --host 127.0.0.1 --port 13141
   --tp "${TEACHER_NUM_GPUS}"
   --chunked-prefill-size 4096
   --mem-fraction-static "${TEACHER_MEM_FRACTION}"
)
if [[ "${OPD_TYPE}" == mimo ]]; then
   TEACHER_CMD=(
      python3 -m slime.rollout.mimo_teacher
      --teacher-model "${TEACHER_MODEL}"
      --student-model "${STUDENT_MODEL}"
      --save-dir "${SAVE_DIR}"
      --max-context "${ROLLOUT_MAX_CONTEXT_LEN}"
      --host 127.0.0.1 --port 13141
   )
fi

echo "OPD: ${TEACHER_NAME} -> ${STUDENT_NAME}; checkpoints: ${SAVE_DIR}"
echo "Actor/rollout GPUs: ${TRAIN_GPUS}; teacher GPUs: ${TEACHER_GPUS}"
for required_path in "${STUDENT_MODEL}/config.json" "${TEACHER_MODEL}/config.json" \
   "${STUDENT_MCORE}" "${DATA_PATH}"; do
   if [[ ! -e "${required_path}" ]]; then
      echo "Missing required path: ${required_path}" >&2
      exit 1
   fi
done
mkdir -p "${SAVE_DIR}" "$(dirname -- "${TEACHER_LOG_FILE}")"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

TEACHER_PID=""
RAY_TEMP_DIR=""
cleanup() {
   local exit_code=$?
   trap - EXIT
   trap '' INT TERM HUP
   if [[ -n "${RAY_TEMP_DIR}" ]]; then
      python3 "${TRAIN_DIR}/utils/stop_isolated_ray.py" "${RAY_TEMP_DIR}" \
         || echo "Ray cleanup failed; logs: ${RAY_TEMP_DIR}" >&2
   fi
   if [[ -n "${TEACHER_PID}" ]]; then
      kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || true
      for _ in {1..10}; do
         kill -0 -- "-${TEACHER_PID}" 2>/dev/null || break
         sleep 1
      done
      kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || true
      wait "${TEACHER_PID}" 2>/dev/null || true
   fi
   exit "${exit_code}"
}
# Register before launching anything: even a failed `ray start` leaves processes.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Fixed ports, matching Slime's example.
CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" setsid "${TEACHER_CMD[@]}" >"${TEACHER_LOG_FILE}" 2>&1 &
TEACHER_PID=$!
deadline=$((SECONDS + 600))
until curl -sf --connect-timeout 2 --max-time 5 http://127.0.0.1:13141/health_generate >/dev/null; do
   if ! kill -0 "${TEACHER_PID}" 2>/dev/null || (( SECONDS >= deadline )); then
      tail -n 40 "${TEACHER_LOG_FILE}" >&2
      echo "Teacher failed to become ready; see ${TEACHER_LOG_FILE}." >&2
      exit 1
   fi
   sleep 5
done

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
RAY_TEMP_DIR="$(mktemp -d /tmp/sd-opd-ray.XXXXXXXX)"
export SD_OPD_RAY_RUN_ID="${RAY_TEMP_DIR}"
echo "Ray logs: ${RAY_TEMP_DIR}/session_latest/logs"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${TRAIN_GPU_COUNT}" \
   --temp-dir "${RAY_TEMP_DIR}" \
   --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 &
# Waiting on a background command lets INT/TERM/HUP trigger cleanup promptly.
wait "$!"
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
keys = ("PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS", "SD_OPD_RAY_RUN_ID")
print(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))
')
ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_CMD[@]}" 2>&1 | tee "${SAVE_DIR}/train.log" &
wait "$!"
