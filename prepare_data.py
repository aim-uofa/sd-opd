#!/usr/bin/env python3
"""Prepare math eval JSONL with a shared suffix; keep downloaded files unchanged."""

import json
from pathlib import Path

import pyarrow.parquet as pq


PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DATASETS = (
    ("aime24", "aime24/aime-2024.jsonl", "prompt", "label"),
    ("aime25", "aime25/aime-2025.jsonl", "prompt", "label"),
    ("hmmt25", "hmmt25/data/train-00000-of-00001.parquet", "problem", "answer"),
    ("amc23", "amc23/test-00000-of-00001.parquet", "question", "answer"),
    ("aime26", "aime26/data/train-00000-of-00001.parquet", "problem", "answer"),
    ("math500", "math500/test.jsonl", "problem", "answer"),
)


def prepare(data_dir: Path) -> None:
    output_dir = data_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, relative_path, input_key, label_key in DATASETS:
        source = data_dir / relative_path
        if source.suffix == ".parquet":
            rows = pq.read_table(source).to_pylist()
        else:
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

        samples = []
        for row in rows:
            prompt = row[input_key]
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            if not messages or messages[-1]["role"] != "user":
                raise ValueError(f"{name}: expected a prompt ending with a user message")
            if not messages[-1]["content"].endswith(PROMPT_SUFFIX):
                messages[-1]["content"] += PROMPT_SUFFIX
            sample = {"prompt": messages, "label": str(row[label_key]).strip()}
            if "metadata" in row:
                sample["metadata"] = row["metadata"]
            samples.append(sample)

        output = output_dir / f"{name}.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8"
        )
        temporary.replace(output)
        print(f"Prepared {output} ({len(samples)} problems)")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parent / "data")
