"""Fast four-dataset evaluation with one shared model/GPU runtime."""

import argparse
import csv
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np
import torch

from transllm.test import run_transllm as single_eval


DATASETS = {
    "SD": {
        "prompt": "data/prompt_data/SD_2021_test.json",
        "st_data": "data/prompt_data/SD_2021_test_pkl.pkl",
        "node_count": 673,
    },
    "SZ": {
        "prompt": "data/prompt_data/SZ_2022_test.json",
        "st_data": "data/prompt_data/SZ_2022_test_pkl.pkl",
        "node_count": 247,
    },
    "pems08": {
        "prompt": "data/prompt_data/pems08_test.json",
        "st_data": "data/prompt_data/pems08_test_pkl.pkl",
        "node_count": 170,
    },
    "urbanev": {
        "prompt": "data/prompt_data/urbanev_test.json",
        "st_data": "data/prompt_data/urbanev_test_pkl.pkl",
        "node_count": 275,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint on small samples from all four test sets "
            "while loading the model and graph resources only once."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        help="Stage 1 adapter checkpoint reconstructed on top of --base-model",
    )
    model_group.add_argument(
        "--model-name",
        help="Complete Hugging Face model, including a Stage 2 checkpoint",
    )
    parser.add_argument("--base-model", default="./checkpoints/llama3-8b")
    parser.add_argument("--output-root", required=True)
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Use the same record count for every dataset (quick evaluation)",
    )
    range_group.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help=(
            "Evaluate this many complete time windows per dataset; the record "
            "count is num-windows multiplied by that dataset's node count"
        ),
    )
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--fixed-prompt-index",
        type=int,
        choices=range(4),
        default=None,
    )
    parser.add_argument("--mape-threshold", type=float, default=1e-5)
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.num_windows is not None and args.num_windows <= 0:
        parser.error("--num-windows must be positive")
    if args.num_samples is None and args.num_windows is None:
        args.num_samples = 12
    if args.start_id < 0:
        parser.error("--start-id must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.mape_threshold < 0:
        parser.error("--mape-threshold must be non-negative")
    return args


def move_graphs_to_cuda(graphs):
    """Cache reusable graph tensors on GPU for all four evaluations."""
    for graph in graphs.values():
        for key in ("nodes_feature", "sp_matrix", "se_matrix"):
            value = graph.get(key)
            if torch.is_tensor(value):
                graph[key] = value.cuda(non_blocking=True)
    return graphs


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def calculate_metrics(result_file, requested_samples, mape_threshold):
    with open(result_file, "r") as handle:
        records = json.load(handle)
    if not records:
        raise RuntimeError(f"No valid predictions in {result_file}")

    predictions = []
    targets = []
    for record in records:
        prediction = np.asarray(record["st_pre_infolow"], dtype=np.float64).reshape(-1)
        target = np.asarray(record["y_in"], dtype=np.float64).reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for {record.get('id')}: "
                f"{prediction.shape} vs {target.shape}"
            )
        predictions.append(prediction)
        targets.append(target)

    # Match metric_calculation/result_test.py and the paper evaluation pipeline.
    prediction = np.abs(np.stack(predictions, axis=0))
    target = np.stack(targets, axis=0)
    absolute_error = np.abs(prediction - target)
    squared_error = np.square(prediction - target)
    denominator_mask = np.abs(target) > mape_threshold

    horizon_metrics = []
    for horizon in range(prediction.shape[1]):
        horizon_mask = denominator_mask[:, horizon]
        horizon_mape = (
            np.mean(
                absolute_error[horizon_mask, horizon]
                / np.abs(target[horizon_mask, horizon])
            )
            * 100.0
            if np.any(horizon_mask)
            else np.nan
        )
        horizon_metrics.append(
            {
                "horizon": horizon + 1,
                "mae": finite_float(np.mean(absolute_error[:, horizon])),
                "rmse": finite_float(np.sqrt(np.mean(squared_error[:, horizon]))),
                "mape_percent": finite_float(horizon_mape),
            }
        )

    all_valid_mape = (
        np.mean(absolute_error[denominator_mask] / np.abs(target[denominator_mask]))
        * 100.0
        if np.any(denominator_mask)
        else np.nan
    )
    return {
        "requested_records": requested_samples,
        "valid_records": len(records),
        "invalid_records": requested_samples - len(records),
        "prediction_steps": int(prediction.shape[1]),
        "mape_threshold": mape_threshold,
        "average": {
            "mae": finite_float(np.mean(absolute_error)),
            "rmse": finite_float(np.sqrt(np.mean(squared_error))),
            "mape_percent": finite_float(all_valid_mape),
        },
        "horizons": horizon_metrics,
    }


def build_eval_args(
    args, dataset, config, output_dir, end_id, requested_samples
):
    return SimpleNamespace(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        fixed_prompt_index=args.fixed_prompt_index,
        prompting_file=config["prompt"],
        conv_mode=None,
        st_data_path=config["st_data"],
        output_res_path=output_dir,
        num_gpus=1,
        max_new_tokens=args.max_new_tokens,
        start_id=args.start_id,
        end_id=end_id,
        num_samples=requested_samples,
        dataset=dataset,
    )


def write_summary_csv(summary, output_file):
    fieldnames = [
        "dataset",
        "requested_records",
        "valid_records",
        "invalid_records",
        "mae",
        "rmse",
        "mape_percent",
        "elapsed_seconds",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, result in summary["datasets"].items():
            metrics = result["metrics"]
            writer.writerow(
                {
                    "dataset": dataset,
                    "requested_records": metrics["requested_records"],
                    "valid_records": metrics["valid_records"],
                    "invalid_records": metrics["invalid_records"],
                    "mae": metrics["average"]["mae"],
                    "rmse": metrics["average"]["rmse"],
                    "mape_percent": metrics["average"]["mape_percent"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Four-dataset checkpoint evaluation requires CUDA")

    if args.checkpoint is not None:
        args.checkpoint = osp.abspath(args.checkpoint)
    if args.model_name is not None:
        args.model_name = osp.abspath(args.model_name)
    args.base_model = osp.abspath(args.base_model)
    args.output_root = osp.abspath(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)

    missing_inputs = [
        path
        for config in DATASETS.values()
        for path in (config["prompt"], config["st_data"])
        if not osp.isfile(path)
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing_inputs}")

    load_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        model_name=args.model_name,
        fixed_prompt_index=args.fixed_prompt_index,
    )
    single_eval.disable_torch_init()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer, model = single_eval.load_evaluation_model(load_args)
    model = model.to("cuda")
    model.get_st_tower().to(device="cuda", dtype=torch.float32)
    model.eval()

    graph_args = single_eval.get_config()
    graph_args.bs = 1
    graphs = move_graphs_to_cuda(single_eval.load_adj(graph_args))
    load_seconds = time.perf_counter() - load_started
    print(f"shared_runtime_loaded_seconds: {load_seconds:.3f}")

    summary = {
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "base_model": args.base_model,
        "start_id": args.start_id,
        "num_samples_per_dataset": args.num_samples,
        "num_windows_per_dataset": args.num_windows,
        "fixed_prompt_index": args.fixed_prompt_index,
        "max_new_tokens": args.max_new_tokens,
        "shared_runtime_load_seconds": round(load_seconds, 3),
        "datasets": {},
    }

    for dataset, config in DATASETS.items():
        prompts = single_eval.load_prompting_file(config["prompt"])
        if args.start_id >= len(prompts):
            raise ValueError(
                f"{dataset}: start-id {args.start_id} is outside "
                f"[0, {len(prompts)})"
            )
        requested_samples = (
            args.num_windows * config["node_count"]
            if args.num_windows is not None
            else args.num_samples
        )
        if args.num_windows is not None and args.start_id % config["node_count"]:
            raise ValueError(
                f"{dataset}: start-id {args.start_id} is not aligned to its "
                f"{config['node_count']} records per complete time window"
            )
        end_id = min(args.start_id + requested_samples, len(prompts))
        if end_id - args.start_id != requested_samples:
            raise ValueError(
                f"{dataset}: requested {requested_samples} records but only "
                f"{end_id - args.start_id} are available"
            )
        selected_prompts = prompts[args.start_id:end_id]
        output_dir = osp.join(args.output_root, dataset)
        os.makedirs(output_dir, exist_ok=True)
        eval_args = build_eval_args(
            args, dataset, config, output_dir, end_id, requested_samples
        )

        print(
            f"\n===== {dataset}: records {args.start_id}:{end_id} "
            f"(model reused) ====="
        )
        started = time.perf_counter()
        single_eval.eval_model(
            eval_args,
            selected_prompts,
            args.start_id,
            end_id,
            tokenizer=tokenizer,
            model=model,
            graphs=graphs,
        )
        elapsed_seconds = time.perf_counter() - started
        result_file = osp.join(
            output_dir,
            f"arxiv_test_res_{args.start_id}_{end_id}.json",
        )
        metrics = calculate_metrics(
            result_file,
            requested_samples=len(selected_prompts),
            mape_threshold=args.mape_threshold,
        )
        with open(osp.join(output_dir, "metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)

        summary["datasets"][dataset] = {
            "prompt_file": osp.abspath(config["prompt"]),
            "st_data_file": osp.abspath(config["st_data"]),
            "result_file": result_file,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "metrics": metrics,
        }
        average = metrics["average"]
        print(
            f"{dataset} quick metrics: "
            f"MAE={average['mae']:.4f}, "
            f"RMSE={average['rmse']:.4f}, "
            f"MAPE={average['mape_percent']:.4f}%, "
            f"valid={metrics['valid_records']}/"
            f"{metrics['requested_records']}, "
            f"elapsed={elapsed_seconds:.2f}s"
        )

    summary_file = osp.join(args.output_root, "summary.json")
    with open(summary_file, "w") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(summary, osp.join(args.output_root, "summary.csv"))
    print(f"\nsummary_json: {summary_file}")
    print(f"summary_csv: {osp.join(args.output_root, 'summary.csv')}")


if __name__ == "__main__":
    main()
