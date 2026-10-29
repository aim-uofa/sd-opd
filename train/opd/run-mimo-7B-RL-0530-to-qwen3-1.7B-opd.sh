#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_NAME=MiMo-7B-RL-0530
export STUDENT_NAME=Qwen3-1.7B
export OPD_TYPE=mimo
exec bash "${SCRIPT_DIR}/run_opd.sh" "$@"
