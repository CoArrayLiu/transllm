#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/transllm/bin/python}"
MODEL_NAME="${1:-./checkpoints/transllm_4dataset/stage2_router_from_70000_normreward/checkpoint-10000}"
OUTPUT_ROOT="${2:-./result_checkpoint/stage2_ckpt10000/zero_shot_fixed_seed}"
SEED="${SEED:-42}"
WINDOWS="${WINDOWS:-12}"
WORKERS="${WORKERS:-2}"
NODE_COUNT="${NODE_COUNT:-170}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-150}"
SUBSET_ROOT="${SUBSET_ROOT:-./data/prompt_data/zero_shot_fixed_seed}"

for value_name in SEED WINDOWS WORKERS NODE_COUNT MAX_NEW_TOKENS LAUNCH_DELAY_SECONDS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer" >&2
        exit 1
    fi
done
if (( WINDOWS == 0 || WORKERS == 0 || NODE_COUNT == 0 || MAX_NEW_TOKENS == 0 )); then
    echo "WINDOWS, WORKERS, NODE_COUNT and MAX_NEW_TOKENS must be positive" >&2
    exit 1
fi

cd "${ROOT_DIR}"
total_records=$((WINDOWS * NODE_COUNT))
if (( WORKERS > total_records )); then
    WORKERS="${total_records}"
fi

result_is_complete() {
    local result_file="$1"
    local expected_count="$2"
    [[ -f "${result_file}" ]] || return 1
    "${PYTHON_BIN}" -c         'import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1]))) == int(sys.argv[2]) else 1)'         "${result_file}" "${expected_count}"         >/dev/null 2>&1
}

for dataset in pems03 pems04; do
    source_prompt="./data/prompt_data/${dataset}_zeroshot.json"
    st_data_file="./data/prompt_data/${dataset}_zeroshot_pkl.pkl"
    subset_dir="${SUBSET_ROOT}/seed_${SEED}_windows_${WINDOWS}"
    subset_prompt="${subset_dir}/${dataset}.json"
    manifest="${subset_dir}/${dataset}_manifest.json"
    output_dir="${OUTPUT_ROOT}/seed_${SEED}_windows_${WINDOWS}/${dataset}"

    for required_file in         "${source_prompt}"         "${st_data_file}"         "./data/st_data/${dataset}/${dataset}_adj_clip.npy"         "./data/st_data/${dataset}/cached_dist_matrix.npy"; do
        if [[ ! -f "${required_file}" ]]; then
            echo "Missing required file: ${required_file}" >&2
            exit 1
        fi
    done

    "${PYTHON_BIN}" scripts/build_fixed_seed_prompt_subset.py         --input-prompt "${source_prompt}"         --output-prompt "${subset_prompt}"         --manifest "${manifest}"         --windows "${WINDOWS}"         --seed "${SEED}"         --expected-nodes "${NODE_COUNT}"

    mkdir -p "${output_dir}"
    echo "Evaluating ${dataset}: seed=${SEED}, windows=${WINDOWS}, records=${total_records}"

    base_count=$((total_records / WORKERS))
    remainder=$((total_records % WORKERS))
    start_id=0
    launched=0
    pids=()
    logs=()

    for ((worker = 0; worker < WORKERS; worker++)); do
        sample_count="${base_count}"
        if (( worker < remainder )); then
            sample_count=$((sample_count + 1))
        fi
        end_id=$((start_id + sample_count))
        result_file="${output_dir}/arxiv_test_res_${start_id}_${end_id}.json"
        log_file="${output_dir}/worker_${start_id}_${end_id}.log"

        if result_is_complete "${result_file}" "${sample_count}"; then
            echo "  complete, skipping [${start_id}, ${end_id})"
            start_id="${end_id}"
            continue
        fi

        if (( launched > 0 && LAUNCH_DELAY_SECONDS > 0 )); then
            echo "  waiting ${LAUNCH_DELAY_SECONDS}s before loading the next model"
            sleep "${LAUNCH_DELAY_SECONDS}"
        fi

        echo "  launching worker ${worker}: [${start_id}, ${end_id})"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}"             -m transllm.test.run_transllm             --model-name "${MODEL_NAME}"             --prompting_file "${subset_prompt}"             --st_data_path "${st_data_file}"             --output_res_path "${output_dir}"             --start_id "${start_id}"             --num-samples "${sample_count}"             --max_new_tokens "${MAX_NEW_TOKENS}"             --num_gpus 1             >"${log_file}" 2>&1 &

        pids+=("$!")
        logs+=("${log_file}")
        launched=$((launched + 1))
        start_id="${end_id}"
    done

    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[${index}]}"; then
            echo "Worker failed: ${logs[${index}]}" >&2
            tail -n 40 "${logs[${index}]}" >&2
            failed=1
        fi
    done
    if (( failed != 0 )); then
        exit 1
    fi

    "${PYTHON_BIN}" -m metric_calculation.result_test         --folder_path "${output_dir}"         --dataset "${dataset}"         | tee "${output_dir}/metrics.log"
done

