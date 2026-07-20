#!/usr/bin/env python3
"""Prepare the 170-node PEMS03/PEMS04 inputs used for zero-shot evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_sensor_ids(path: Path) -> list[str]:
    sensor_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError(f"duplicate sensor ids in {path}")
    return sensor_ids


def build_adjacency(csv_path: Path, selected_ids: list[str]) -> tuple[np.ndarray, int]:
    id_to_index = {sensor_id: index for index, sensor_id in enumerate(selected_ids)}
    adjacency = np.zeros((len(selected_ids), len(selected_ids)), dtype=np.float32)
    retained_edges = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"from", "to"}.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain 'from' and 'to' columns")
        for row in reader:
            source = str(row["from"]).strip()
            target = str(row["to"]).strip()
            if source not in id_to_index or target not in id_to_index:
                continue
            source_index = id_to_index[source]
            target_index = id_to_index[target]
            adjacency[source_index, target_index] = 1.0
            adjacency[target_index, source_index] = 1.0
            retained_edges += 1
    if not np.count_nonzero(adjacency):
        raise ValueError(f"no edges remain after clipping {csv_path}")
    return adjacency, retained_edges


def prepare_dataset(root: Path, name: str, node_count: int, overwrite: bool) -> None:
    dataset_dir = root / "data" / "st_data" / name
    upper_name = name.upper()
    raw_path = dataset_dir / f"{upper_name}.npz"
    csv_path = dataset_dir / f"{upper_name}.csv"
    clip_path = dataset_dir / f"{name}_clip.npz"
    adjacency_path = dataset_dir / f"{name}_adj_clip.npy"
    for path in (raw_path, csv_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not overwrite and (clip_path.exists() or adjacency_path.exists()):
        raise FileExistsError(
            f"prepared files already exist for {name}; pass --overwrite to replace them"
        )
    with np.load(raw_path) as archive:
        if "data" not in archive:
            raise KeyError(f"{raw_path} does not contain a 'data' array")
        data = np.asarray(archive["data"])
    if data.ndim != 3 or data.shape[1] < node_count:
        raise ValueError(
            f"{raw_path}: expected [time, nodes, features] with at least "
            f"{node_count} nodes, got {data.shape}"
        )
    if name == "pems03":
        id_path = dataset_dir / "PEMS03.txt"
        if not id_path.is_file():
            raise FileNotFoundError(
                f"{id_path} is required to map PEMS03 array columns to sensor ids"
            )
        all_ids = read_sensor_ids(id_path)
        if len(all_ids) != data.shape[1]:
            raise ValueError(
                f"{id_path} has {len(all_ids)} ids but {raw_path} has "
                f"{data.shape[1]} nodes"
            )
        selected_ids = all_ids[:node_count]
    else:
        # PEMS04.csv uses zero-based array-column indices directly.
        selected_ids = [str(index) for index in range(node_count)]
    clipped = data[:, :node_count, :]
    adjacency, retained_edges = build_adjacency(csv_path, selected_ids)
    np.savez_compressed(clip_path, data=clipped)
    np.save(adjacency_path, adjacency)
    print(
        f"[{name}] data {data.shape} -> {clipped.shape}; "
        f"adjacency={adjacency.shape}, raw_edges_retained={retained_edges}, "
        f"directed_nonzeros={np.count_nonzero(adjacency)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--datasets", nargs="+", choices=("pems03", "pems04"),
        default=("pems03", "pems04"),
    )
    parser.add_argument("--node-count", type=int, default=170)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.node_count <= 0:
        raise ValueError("--node-count must be positive")
    for dataset in args.datasets:
        prepare_dataset(args.root.resolve(), dataset, args.node_count, args.overwrite)


if __name__ == "__main__":
    main()
