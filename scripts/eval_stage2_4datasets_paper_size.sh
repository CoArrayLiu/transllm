#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_stage2_4datasets_paper_size.sh [MODEL_PATH] [OUTPUT_ROOT]

Defaults:
  MODEL_PATH   ./checkpoints/transllm_4dataset/stage2_router_from_70000_normreward/checkpoint-10000
  OUTPUT_ROOT  ./result_checkpoint/stage2_router_from_70000_normreward/checkpoint-10000/paper_12windows

The model is loaded once, then each local dataset is evaluated on 12 complete
time windows: SD=673*12, SZ=247*12, PEMS08=170*12, UrbanEV=275*12.

Optional environment variables:
  CUDA_DEVICE       CUDA_VISIBLE_DEVICES value (default: 0)
  MAX_NEW_TOKENS    Generation safety limit (default: 128)
  MAPE_THRESHOLD    Ignore |target| <= threshold in MAPE (default: 1e-5)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

model_path=${1:-./checkpoints/transllm_4dataset/stage2_router_from_70000_normreward/checkpoint-10000}
output_root=${2:-./result_checkpoint/stage2_router_from_70000_normreward/checkpoint-10000/paper_12windows}
cuda_device=${CUDA_DEVICE:-0}
max_new_tokens=${MAX_NEW_TOKENS:-128}
mape_threshold=${MAPE_THRESHOLD:-1e-5}

if [[ ! -d "$model_path" ]]; then
  echo "Stage 2 model directory not found: $model_path" >&2
  exit 2
fi
if [[ -e "$output_root" && -n "$(find "$output_root" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty; use a new directory: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"

command_args=(
  python -m transllm.test.run_four_dataset_eval
  --model-name "$model_path"
  --output-root "$output_root"
  --num-windows 12
  --start-id 0
  --max-new-tokens "$max_new_tokens"
  --mape-threshold "$mape_threshold"
)

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$cuda_device"
  printf '%q ' "${command_args[@]}"
  printf '\n'
} > "$output_root/command.txt"

echo "Model: $model_path"
echo "Output: $output_root"
echo "Records: SD=8076, SZ=2964, pems08=2040, urbanev=3300"
echo "The Stage 2 model and graph tensors will be loaded once."

CUDA_VISIBLE_DEVICES="$cuda_device" "${command_args[@]}" 2>&1 \
  | tee "$output_root/run.log"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" \
  >> "$output_root/command.txt"
echo "Summary JSON: $output_root/summary.json"
echo "Summary CSV:  $output_root/summary.csv"
