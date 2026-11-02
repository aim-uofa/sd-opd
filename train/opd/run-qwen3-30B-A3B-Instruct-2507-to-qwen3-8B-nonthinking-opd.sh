#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_NAME=Qwen3-30B-A3B-Instruct-2507
export STUDENT_NAME=Qwen3-8B
export TP_SIZE="${TP_SIZE:-2}"
exec bash "${SCRIPT_DIR}/../run_opd.sh" "$@"
