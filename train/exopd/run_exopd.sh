#!/usr/bin/env bash
# ExOPD uses the frozen initial student and the common OPD training settings.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPD_METHOD=exopd
export EXOPD_SCALE="${EXOPD_SCALE:-1.25}"

# Keep the initial reference fixed, including when resuming the actor.
for arg in "$@"; do
   case "${arg%%=*}" in
      --load|--save|--hf-checkpoint|--ref-load|--ref-ckpt-step|--ref-update-interval|--opd-type)
         echo "${arg%%=*} is managed by ExOPD; use SAVE_DIR, STUDENT_MODEL or STUDENT_MCORE." >&2
         exit 1
         ;;
   esac
done
exec bash "${SCRIPT_DIR}/../opd/run_opd.sh" "$@"
