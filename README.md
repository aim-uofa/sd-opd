# sd-opd

On-policy distillation and math evaluation scripts built on [slime](https://github.com/THUDM/slime).

## Environment setup

Follow the Conda environment setup in the [slime installation guide](https://thudm.github.io/slime/get_started/quick_start.html).

## Prepare data

```bash
bash download_data.sh
```

This downloads DAPO-Math-17k and six evaluation datasets: AIME 2024, AIME 2025, AIME 2026, HMMT February 2025, AMC 2023, and MATH-500. It then runs [prepare_data.py](prepare_data.py) to write the evaluation files to `data/eval/`, keeping the original downloads in `data/`.

All six evaluation datasets use the same suffix:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

The shared evaluation configuration is [eval/math_multi_task.yaml](eval/math_multi_task.yaml).

## OPD training

The four vanilla OPD entries in [train/opd/](train/opd/) share the training hyperparameters in [train/config.sh](train/config.sh) and the Slime launcher in [train/opd/run_opd.sh](train/opd/run_opd.sh).

| Teacher | Student | Script |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507 | Qwen3-1.7B | [4B-Instruct → 1.7B](train/opd/run-qwen3-4B-Instruct-2507-to-qwen3-1.7B-nonthinking-opd.sh) |
| Qwen3-30B-A3B-Instruct-2507 | Qwen3-1.7B | [30B-A3B → 1.7B](train/opd/run-qwen3-30B-A3B-Instruct-2507-to-qwen3-1.7B-nonthinking-opd.sh) |
| Qwen3-30B-A3B-Instruct-2507 | Qwen3-4B | [30B-A3B → 4B](train/opd/run-qwen3-30B-A3B-Instruct-2507-to-qwen3-4B-nonthinking-opd.sh) |
| Qwen3-30B-A3B-Instruct-2507 | Qwen3-8B | [30B-A3B → 8B](train/opd/run-qwen3-30B-A3B-Instruct-2507-to-qwen3-8B-nonthinking-opd.sh) |

Defaults follow `pilot_train` vanilla OPD: 272 rollout rounds, 64 prompts × 4 responses, global batch 64, prompt/response limits 1024/7168, temperature/top-p 1, constant learning rate `1e-6`, Adam betas 0.9/0.98, weight decay 0.1, PPO clip 0.2, token-mean loss, seed 11, and checkpoint saves every 20 rounds plus the final round. Students use `enable_thinking=false`; Slime's SGLang OPD reward hook scores the same token sequence with the teacher. Training does not run evaluation automatically.

Prepare the data with `bash download_data.sh` and place the Hugging Face models under `models/`, or set `WEIGHTS_DIR` to an existing model root. For example:

```bash
WEIGHTS_DIR=/mnt/cpfs/weights \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash train/opd/run-qwen3-4B-Instruct-2507-to-qwen3-1.7B-nonthinking-opd.sh
```

All four entries default to 4 GPUs (actor/rollout/teacher = 2/1/1) and 16384-token training/log-prob budgets. The Qwen3-8B student uses actor TP=2 with sequence parallelism; the other students use TP=1 by default. The shared `--make-vocab-size-divisible-by 64` keeps the 151936-token vocabulary unchanged at both TP sizes. All entries accept 4 or 8 GPUs via `CUDA_VISIBLE_DEVICES`; 8 GPUs use a 2/4/2 allocation, with the final two GPUs serving the teacher at TP=2 as in Slime's example. Teacher memory fraction defaults to Slime's 0.8.

The dynamic token budget controls microbatch packing without changing global batch size, optimizer updates per rollout, or response length. Parallelism and packing can change floating-point results; identical training scores are not guaranteed. GPU memory fit must be checked on the target hardware.

Edit `train/config.sh` or override its variables, such as `LR=5e-7 NUM_ROLLOUT=20`. `STUDENT_MODEL`, `TEACHER_MODEL`, `DATA_PATH`, and `SAVE_DIR` override individual paths; extra command-line arguments are passed to Slime. Students initialize from Hugging Face weights, or from a converted checkpoint via `STUDENT_MCORE`, and resume from `SAVE_DIR` when it contains a checkpoint. Each setting saves to its own `checkpoints/<teacher>-to-<student>-opd-nonthinking-seed-<seed>/` directory, including `train.log` and `teacher.log`.

Use `DRY_RUN=1 bash train/opd/<script>.sh` to print both commands without starting services or requiring model weights. Teacher and Ray dashboard ports are fixed at 13141 and 8265; there is no port selection or per-setting port offset. The teacher and local Ray instance are stopped when the script exits. If an earlier run left Ray running, run `ray stop` once before restarting.

## Evaluation

Both evaluation scripts enable SGLang deterministic inference and use `--rollout-seed 42`.

### Hugging Face models

For example, download and evaluate Qwen3-4B:

```bash
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
CUDA_VISIBLE_DEVICES=0,1 bash eval/eval_qwen3.sh qwen3-4b
```

The script looks for models in `models/<model name>`. Set `HF_CHECKPOINT` to use an existing local model directory.

| Script argument | Model name | Default GPUs per engine |
| --- | --- | --- |
| `qwen3-1.7b` | `Qwen3-1.7B` | 1 |
| `qwen3-4b` | `Qwen3-4B` | 2 |
| `qwen3-8b` | `Qwen3-8B` | 2 |
| `qwen3-4b-instruct-2507` | `Qwen3-4B-Instruct-2507` | 2 |
| `qwen3-30b-a3b-instruct-2507` | `Qwen3-30B-A3B-Instruct-2507` | 8 |

Use `CUDA_VISIBLE_DEVICES` to select GPUs. Their count must be a multiple of `TP_SIZE`, which overrides the GPUs per engine shown above. Results are saved under `eval/results/`. Both evaluation scripts stop their local Ray instance on exit. If an earlier run left Ray running, run `ray stop` once before restarting evaluation.

### Megatron checkpoints

Set `BASE_MODEL` to the matching original Hugging Face model directory and `CHECKPOINT_PATH` to the directory containing `iter_XXXXXXX` checkpoints:

```bash
BASE_MODEL=models/Qwen3-4B \
CHECKPOINT_PATH=/path/to/checkpoints \
CHECKPOINT_STEP=119 \
CUDA_VISIBLE_DEVICES=0,1 \
bash eval/eval_checkpoint.sh qwen3-4b
```

This converts `iter_0000119` to Hugging Face format with slime's conversion tool, caches it under `CHECKPOINT_PATH/hf/iter_0000119`, and runs the same evaluation configuration.
