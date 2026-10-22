#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "$DATA_DIR"

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "$DATA_DIR/dapo-math-17k"

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "$DATA_DIR/aime24"

hf download --repo-type dataset zhuzilin/aime-2025 \
  --local-dir "$DATA_DIR/aime25"

hf download --repo-type dataset MathArena/hmmt_feb_2025 \
  --local-dir "$DATA_DIR/hmmt25"

hf download --repo-type dataset math-ai/amc23 \
  --local-dir "$DATA_DIR/amc23"

hf download --repo-type dataset MathArena/aime_2026 \
  --revision d61f486805306e6e6b5913504c07763a99e1b112 \
  --local-dir "$DATA_DIR/aime26"

hf download --repo-type dataset HuggingFaceH4/MATH-500 \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --local-dir "$DATA_DIR/math500"

python3 "${SCRIPT_DIR}/prepare_data.py"
