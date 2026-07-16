#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-8}"

cd "${ROOT_DIR}"

if [[ ! -d data/st_data/sd || ! -d data/st_data/shenzhen ]]; then
    echo "Dataset paths are not normalized. Run scripts/normalize_dataset_paths.sh first." >&2
    exit 1
fi

"${PYTHON_BIN}" -c 'import fastdtw, h5py, joblib, numpy' >/dev/null
"${PYTHON_BIN}" scripts/generate_cache_matrices.py --jobs "${JOBS}" "$@"
