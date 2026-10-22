# sd-opd

Math evaluation scripts built on [slime](https://github.com/THUDM/slime).

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

Use `CUDA_VISIBLE_DEVICES` to select GPUs. Their count must be a multiple of `TP_SIZE`, which overrides the GPUs per engine shown above. Results are saved under `eval/results/`. After an evaluation finishes, run `ray stop` to stop its local Ray instance before starting another evaluation.

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
