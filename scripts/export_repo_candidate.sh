#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/opm-source-roi" >&2
    exit 2
fi

target_dir="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "$script_dir/.." && pwd)"

mkdir -p "$target_dir"

rsync -a \
    --exclude '.git/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.py[cod]' \
    --exclude '.DS_Store' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info/' \
    --exclude '.venv/' \
    "$source_dir/" "$target_dir/"

echo "Standalone repo candidate exported to: $target_dir"
echo "Next steps:"
echo "  cd \"$target_dir\""
echo "  git init"
echo "  uv pip install -e \".[surface,alignment-qc,dev]\""
echo "  python -m pytest -q tests/test_opm_source_toolbox.py"
