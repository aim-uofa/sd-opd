#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_NAME=Qwen3-4B-Instruct-2507
export STUDENT_NAME=Qwen3-1.7B
export OPD_TYPE=sglang
exec bash "${SCRIPT_DIR}/run_exopd.sh" "$@"
