#!/usr/bin/env python3
"""Fast, dependency-free validation for the four-dataset configuration."""

from pathlib import Path
import ast
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = {"SD", "SZ", "pems08", "urbanev"}
REQUIRED_FILES = [
    "checkpoints/pretrained_encoder/st_encoder.pt",
    "data/st_data/sd/sd_rn_adj.npy",
    "data/st_data/sd/cached_dist_matrix.npy",
    "data/st_data/shenzhen/shenzhen_adj.npy",
    "data/st_data/shenzhen/cached_dist_matrix.npy",
    "data/st_data/pems08/pems08_adj.npy",
    "data/st_data/pems08/cached_dist_matrix.npy",
    "data/st_data/urbanev/urbanev_adj.npy",
    "data/st_data/urbanev/cached_dist_matrix.npy",
]
PROMPT_FILES = [
    f"data/prompt_data/{name}_{split}.{suffix}"
    for name in ("SD_2021", "SZ_2022", "pems08", "urbanev")
    for split, suffix in (("supervised", "json"), ("supervised_pkl", "pkl"), ("test", "json"), ("test_pkl", "pkl"))
]


def fail(message):
    print(f"FAIL: {message}")
    return False


def main():
    ok = True
    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            ok = fail(f"missing required file: {relative_path}")

    source_paths = [
        "transllm/train/train_learning_prompt_5dataset.py",
        "transllm/train/train_st_learning_prompt_5dataset.py",
        "transllm/model/STLlama_learning_prompt_5dataset.py",
        "transllm/test/run_transllm.py",
        "instruction_generate/instruction_generate.py",
        "metric_calculation/result_test.py",
    ]
    for relative_path in source_paths:
        try:
            ast.parse((ROOT / relative_path).read_text(), filename=relative_path)
        except SyntaxError as error:
            ok = fail(f"syntax error in {relative_path}: {error}")

    train_source = (ROOT / "transllm/train/train_st_learning_prompt_5dataset.py").read_text()
    dataset_class = train_source.split("class LazySupervisedDataset_ST", 1)[1].split("@dataclass\nclass DataCollator", 1)[0]
    for forbidden in ("data_path_sh", "st_data_path_sh", "square_grid_3km_shanghai", "sh_list_data_dict"):
        if forbidden in dataset_class:
            ok = fail(f"runtime SH dependency remains in dataset class: {forbidden}")

    for relative_path in PROMPT_FILES:
        if not (ROOT / relative_path).is_file():
            print(f"WARN: prompt artifact not generated yet: {relative_path}")

    print(f"Expected runtime datasets: {sorted(EXPECTED_DATASETS)}")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
