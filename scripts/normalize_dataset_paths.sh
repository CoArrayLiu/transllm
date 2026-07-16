#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/st_data"

rename_dataset_dir() {
    local old_name="$1"
    local new_name="$2"
    local old_path="${DATA_DIR}/${old_name}"
    local new_path="${DATA_DIR}/${new_name}"

    if [[ -d "${new_path}" && ! -L "${new_path}" ]]; then
        if [[ -e "${old_path}" ]]; then
            echo "Both paths exist; refusing to merge them:" >&2
            echo "  ${old_path}" >&2
            echo "  ${new_path}" >&2
            exit 1
        fi
        echo "Already normalized: ${new_path}"
        return
    fi

    if [[ -L "${new_path}" ]]; then
        echo "A symbolic link already occupies ${new_path}; remove it first." >&2
        exit 1
    fi

    if [[ ! -d "${old_path}" ]]; then
        echo "Missing source dataset directory: ${old_path}" >&2
        exit 1
    fi

    mv -- "${old_path}" "${new_path}"
    echo "Renamed: ${old_path} -> ${new_path}"
}

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Missing data directory: ${DATA_DIR}" >&2
    exit 1
fi

rename_dataset_dir "SD" "sd"
rename_dataset_dir "SZ" "shenzhen"

echo "Dataset paths now match the paths used by training and evaluation code."
