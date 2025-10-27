#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-$HOME/.venvs/bpref}"

python3 -m venv "$TARGET"
source "$TARGET/bin/activate" || source activate "$TARGET" || true
python -m pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

python -V
pip list | wc -l
echo "Venv ready at: $TARGET"
