#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-llm}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}/checkpoints/llama3-8b}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT_DIR}/checkpoints/transllm_4dataset/stage1_llm}"
STAGE2_DIR="${STAGE2_DIR:-${ROOT_DIR}/checkpoints/transllm_4dataset/stage2_router}"
BITS="${BITS:-8}"

cd "${ROOT_DIR}"

required_prompt_files=(
    data/prompt_data/SD_2021_supervised.json
    data/prompt_data/SD_2021_supervised_pkl.pkl
    data/prompt_data/SZ_2022_supervised.json
    data/prompt_data/SZ_2022_supervised_pkl.pkl
    data/prompt_data/pems08_supervised.json
    data/prompt_data/pems08_supervised_pkl.pkl
    data/prompt_data/urbanev_supervised.json
    data/prompt_data/urbanev_supervised_pkl.pkl
)

for path in "${required_prompt_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${path}. Run ./scripts/generate_prompt_data.sh first." >&2
        exit 1
    fi
done

if [[ ! -f checkpoints/pretrained_encoder/st_encoder.pt ]]; then
    echo "Missing checkpoints/pretrained_encoder/st_encoder.pt" >&2
    exit 1
fi

case "${STAGE}" in
    llm)
        MODEL_PATH="${BASE_MODEL}"
        OUTPUT_DIR="${STAGE1_DIR}"
        ;;
    router)
        MODEL_PATH="${STAGE1_DIR}/full_model"
        OUTPUT_DIR="${STAGE2_DIR}"
        if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
            echo "Stage 1 model not found at ${MODEL_PATH}. Train the llm stage first." >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 {llm|router}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found at ${MODEL_PATH}/config.json" >&2
    exit 1
fi

exec "${PYTHON_BIN}" transllm/train/train_learning_prompt_5dataset.py \
    --training_stage "${STAGE}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --bits "${BITS}" \
    "${@:2}"
