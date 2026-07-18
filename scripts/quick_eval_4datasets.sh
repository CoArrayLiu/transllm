#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/quick_eval_4datasets.sh CHECKPOINT [NUM_SAMPLES] [OUTPUT_ROOT]

Examples:
  scripts/quick_eval_4datasets.sh \
    ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-40000

  FIXED_PROMPT_INDEX=3 MAX_NEW_TOKENS=128 CUDA_DEVICE=0 \
    scripts/quick_eval_4datasets.sh \
    ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-40000 12

Optional environment variables:
  BASE_MODEL          Base Llama path (default: ./checkpoints/llama3-8b)
  CUDA_DEVICE         CUDA_VISIBLE_DEVICES value (default: 0)
  FIXED_PROMPT_INDEX  Empty means use checkpoint routers; otherwise 0..3
  MAX_NEW_TOKENS      Generation safety limit (default: 128)
  START_ID            First record in each test set (default: 0)
  MAPE_THRESHOLD      Ignore |target| <= threshold in MAPE (default: 1e-5)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

checkpoint=$1
num_samples=${2:-12}
base_model=${BASE_MODEL:-./checkpoints/llama3-8b}
cuda_device=${CUDA_DEVICE:-0}
fixed_prompt_index=${FIXED_PROMPT_INDEX:-}
max_new_tokens=${MAX_NEW_TOKENS:-128}
start_id=${START_ID:-0}
mape_threshold=${MAPE_THRESHOLD:-1e-5}

if [[ ! -d "$checkpoint" ]]; then
  echo "Checkpoint directory not found: $checkpoint" >&2
  exit 2
fi
if [[ ! "$num_samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SAMPLES must be a positive integer: $num_samples" >&2
  exit 2
fi
if [[ -n "$fixed_prompt_index" && ! "$fixed_prompt_index" =~ ^[0-3]$ ]]; then
  echo "FIXED_PROMPT_INDEX must be empty or one of 0,1,2,3" >&2
  exit 2
fi

checkpoint_name=$(basename "${checkpoint%/}")
timestamp=$(date +%Y%m%d_%H%M%S)
output_root=${3:-"./result_checkpoint/${checkpoint_name}/quick_4dataset_${timestamp}"}
mkdir -p "$output_root"

command_args=(
  python -m transllm.test.run_four_dataset_eval
  --checkpoint "$checkpoint"
  --base-model "$base_model"
  --output-root "$output_root"
  --num-samples "$num_samples"
  --start-id "$start_id"
  --max-new-tokens "$max_new_tokens"
  --mape-threshold "$mape_threshold"
)
if [[ -n "$fixed_prompt_index" ]]; then
  command_args+=(--fixed-prompt-index "$fixed_prompt_index")
fi

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$cuda_device"
  printf '%q ' "${command_args[@]}"
  printf '\n'
} > "$output_root/command.txt"

echo "Output: $output_root"
echo "The model/checkpoint and all graph tensors will be loaded once."
CUDA_VISIBLE_DEVICES="$cuda_device" "${command_args[@]}" 2>&1 \
  | tee "$output_root/run.log"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" \
  >> "$output_root/command.txt"
echo "Summary: $output_root/summary.csv"
