#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${ROOT_DIR}/data/prompt_data"

cd "${ROOT_DIR}"

if [[ ! -d data/st_data/sd || ! -d data/st_data/shenzhen ]]; then
    echo "Dataset paths are not normalized. Run scripts/normalize_dataset_paths.sh first." >&2
    exit 1
fi

"${PYTHON_BIN}" -c 'import h5py, numpy, pandas, torch' >/dev/null
mkdir -p "${OUTPUT_DIR}"

datasets=(SD_2021 SZ_2022 pems08 urbanev)

for dataset in "${datasets[@]}"; do
    echo "Generating supervised prompt data for ${dataset}..."
    "${PYTHON_BIN}" instruction_generate/instruction_generate.py \
        -dataset_name "${dataset}" \
        -for_zeroshot False \
        -for_supervised True \
        -for_ablation False \
        -for_test False

    echo "Generating test prompt data for ${dataset}..."
    "${PYTHON_BIN}" instruction_generate/instruction_generate.py \
        -dataset_name "${dataset}" \
        -for_zeroshot False \
        -for_supervised False \
        -for_ablation False \
        -for_test True
done

echo "Generated prompt data:"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.pkl' \) -print | sort
