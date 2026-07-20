#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/transllm/bin/python}"
MODEL_NAME="${1:-./checkpoints/transllm_4dataset/stage2_router_from_70000_normreward/checkpoint-10000}"
OUTPUT_ROOT="${2:-./result_checkpoint/stage2_ckpt10000/zero_shot}"
NUM_SAMPLES="${NUM_SAMPLES:-2040}"
WORKERS="${WORKERS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if ! [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SAMPLES must be a positive integer" >&2
    exit 1
fi
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORKERS must be a positive integer" >&2
    exit 1
fi
if (( WORKERS > NUM_SAMPLES )); then
    WORKERS="${NUM_SAMPLES}"
fi

cd "${ROOT_DIR}"

for dataset in pems03 pems04; do
    prompt_file="./data/prompt_data/${dataset}_zeroshot.json"
    st_data_file="./data/prompt_data/${dataset}_zeroshot_pkl.pkl"
    output_dir="${OUTPUT_ROOT}/${dataset}/samples_${NUM_SAMPLES}_workers_${WORKERS}"

    for required_file in \
        "${prompt_file}" \
        "${st_data_file}" \
        "./data/st_data/${dataset}/${dataset}_adj_clip.npy" \
        "./data/st_data/${dataset}/cached_dist_matrix.npy"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "Missing required file: ${required_file}" >&2
            exit 1
        fi
    done
    mkdir -p "${output_dir}"

    echo "Evaluating ${dataset}: ${NUM_SAMPLES} records with ${WORKERS} workers"
    base_count=$((NUM_SAMPLES / WORKERS))
    remainder=$((NUM_SAMPLES % WORKERS))
    start_id=0
    pids=()
    logs=()

    for ((worker = 0; worker < WORKERS; worker++)); do
        sample_count="${base_count}"
        if (( worker < remainder )); then
            sample_count=$((sample_count + 1))
        fi
        end_id=$((start_id + sample_count))
        log_file="${output_dir}/worker_${start_id}_${end_id}.log"

        echo "  worker ${worker}: [${start_id}, ${end_id}) -> ${log_file}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" \
            -m transllm.test.run_transllm \
            --model-name "${MODEL_NAME}" \
            --prompting_file "${prompt_file}" \
            --st_data_path "${st_data_file}" \
            --output_res_path "${output_dir}" \
            --start_id "${start_id}" \
            --num-samples "${sample_count}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --num_gpus 1 \
            >"${log_file}" 2>&1 &

        pids+=("$!")
        logs+=("${log_file}")
        start_id="${end_id}"
    done

    failed=0
    for ((worker = 0; worker < WORKERS; worker++)); do
        if ! wait "${pids[${worker}]}"; then
            echo "Worker failed: ${logs[${worker}]}" >&2
            tail -n 40 "${logs[${worker}]}" >&2
            failed=1
        fi
    done
    if (( failed != 0 )); then
        exit 1
    fi

    "${PYTHON_BIN}" -m metric_calculation.result_test \
        --folder_path "${output_dir}" \
        --dataset "${dataset}" \
        | tee "${output_dir}/metrics.log"
done
