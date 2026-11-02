#!/usr/bin/env bash
# Teacher-gated, centered current-student / frozen-student-base delta.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_METHOD=sd_opd
export STUDENT_BASE_DELTA_TOP_K="${STUDENT_BASE_DELTA_TOP_K:-1024}"
export STUDENT_BASE_DELTA_CHUNK_SIZE="${STUDENT_BASE_DELTA_CHUNK_SIZE:-128}"
export STUDENT_BASE_DELTA_SCORER_TIMEOUT="${STUDENT_BASE_DELTA_SCORER_TIMEOUT:-600}"
export STUDENT_BASE_DELTA_DUAL_CLIP="${STUDENT_BASE_DELTA_DUAL_CLIP:-3.0}"
exec bash "${SCRIPT_DIR}/../run_opd.sh" "$@"
