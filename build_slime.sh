#!/usr/bin/env bash
set -eo pipefail

export BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SLIME_DIR="${BASE_DIR}/slime"
cd "$BASE_DIR"

if conda run -n slime true >/dev/null 2>&1; then
  conda install -n slime python=3.12 pip -c conda-forge -y
else
  conda create -n slime python=3.12 pip -c conda-forge -y
fi

# Keep upstream dependency versions, source checkouts and patches.
# Replace only its micromamba bootstrap and package-install commands.
if ! grep -qx '# create conda' "$SLIME_DIR/build_conda.sh" ||
   ! grep -qx 'micromamba activate slime' "$SLIME_DIR/build_conda.sh"; then
  echo "Unexpected slime/build_conda.sh layout; update the Conda adaptation." >&2
  exit 1
fi
INSTALL_SCRIPT=$(sed \
  -e '/^# create conda$/,/^micromamba activate slime$/d' \
  -e 's/^micromamba install /conda install /' \
  "$SLIME_DIR/build_conda.sh")
if [[ "$INSTALL_SCRIPT" == *micromamba* ]]; then
  echo "Unexpected slime/build_conda.sh layout; update the Conda adaptation." >&2
  exit 1
fi
conda run -n slime --no-capture-output bash -e <<< "$INSTALL_SCRIPT"

echo 'Setup complete. Run: conda activate slime'
