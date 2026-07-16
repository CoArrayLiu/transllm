#!/usr/bin/env python3
"""Generate the DTW-based semantic adjacency matrices used by TransLLM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from fastdtw import fastdtw
from joblib import Parallel, delayed


@dataclass(frozen=True)
class DatasetSpec:
    raw_path: Path
    adjacency_path: Path
    output_path: Path
    points_per_day: int
    loader: Callable[[Path], np.ndarray]


def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["t/block0_values"][:], dtype=np.float64)


def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        data = np.asarray(archive["data"], dtype=np.float64)
    return data[..., 0] if data.ndim == 3 else data


def orient_time_nodes(data: np.ndarray, node_count: int, name: str) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D time series, got {data.shape}")
    if data.shape[1] == node_count:
        return data
    if data.shape[0] == node_count:
        return data.T
    raise ValueError(
        f"{name}: neither axis matches adjacency node count {node_count}: {data.shape}"
    )


def interpolate_nan_columns(data: np.ndarray) -> np.ndarray:
    result = data.copy()
    x = np.arange(result.shape[0])
    for column in range(result.shape[1]):
        values = result[:, column]
        valid = np.isfinite(values)
        if valid.all():
            continue
        if not valid.any():
            raise ValueError(f"node {column} contains no finite observations")
        result[:, column] = np.interp(x, x[valid], values[valid])
    return result


def average_daily_profile(data: np.ndarray, points_per_day: int) -> np.ndarray:
    complete_days = data.shape[0] // points_per_day
    if complete_days == 0:
        raise ValueError(
            f"only {data.shape[0]} points; need at least {points_per_day} for one day"
        )
    trimmed = data[: complete_days * points_per_day]
    # [day, time-of-day, node] -> [node, time-of-day]
    return trimmed.reshape(complete_days, points_per_day, data.shape[1]).mean(0).T


def calculate_distances(profiles: np.ndarray, jobs: int, radius: int) -> np.ndarray:
    node_count = profiles.shape[0]

    def calculate_row(i: int) -> tuple[int, np.ndarray]:
        row = np.empty(node_count - i, dtype=np.float64)
        for offset, j in enumerate(range(i, node_count)):
            row[offset] = fastdtw(profiles[i], profiles[j], radius=radius)[0]
        return i, row

    rows = Parallel(n_jobs=jobs, verbose=10)(
        delayed(calculate_row)(i) for i in range(node_count)
    )
    distances = np.zeros((node_count, node_count), dtype=np.float64)
    for i, row in rows:
        distances[i, i:] = row
        distances[i:, i] = row
    return distances


def threshold_distances(
    distances: np.ndarray, sigma: float, threshold: float
) -> np.ndarray:
    std = distances.std()
    if std == 0:
        raise ValueError("DTW distance matrix has zero standard deviation")
    normalized = (distances - distances.mean()) / std
    similarity = np.exp(-(normalized**2) / (sigma**2))
    return (similarity > threshold).astype(np.uint8)


def build_specs(root: Path) -> dict[str, DatasetSpec]:
    data = root / "data" / "st_data"
    return {
        "sd": DatasetSpec(
            data / "sd" / "sd_his_2021.h5",
            data / "sd" / "sd_rn_adj.npy",
            data / "sd" / "cached_dist_matrix.npy",
            288,
            load_h5,
        ),
        "sz": DatasetSpec(
            data / "shenzhen" / "sz-charge.npz",
            data / "shenzhen" / "shenzhen_adj.npy",
            data / "shenzhen" / "cached_dist_matrix.npy",
            288,
            load_npz,
        ),
        "pems08": DatasetSpec(
            data / "pems08" / "pems08.npz",
            data / "pems08" / "pems08_adj.npy",
            data / "pems08" / "cached_dist_matrix.npy",
            288,
            load_npz,
        ),
        "urbanev": DatasetSpec(
            data / "urbanev" / "urbanev-charge.npz",
            data / "urbanev" / "urbanev_adj.npy",
            data / "urbanev" / "cached_dist_matrix.npy",
            24,
            load_npz,
        ),
    }


def generate_one(
    name: str,
    spec: DatasetSpec,
    jobs: int,
    radius: int,
    sigma: float,
    threshold: float,
    overwrite: bool,
) -> None:
    if spec.output_path.exists() and not overwrite:
        print(f"[{name}] exists, skipping: {spec.output_path}")
        return
    for path in (spec.raw_path, spec.adjacency_path):
        if not path.exists():
            raise FileNotFoundError(f"[{name}] missing required file: {path}")

    adjacency = np.load(spec.adjacency_path, mmap_mode="r")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"[{name}] invalid adjacency shape: {adjacency.shape}")

    raw = spec.loader(spec.raw_path)
    time_nodes = orient_time_nodes(raw, adjacency.shape[0], name)
    time_nodes = interpolate_nan_columns(time_nodes)
    profiles = average_daily_profile(time_nodes, spec.points_per_day)
    print(
        f"[{name}] input={time_nodes.shape}, daily_profile={profiles.shape}, "
        f"pairs={profiles.shape[0] * (profiles.shape[0] + 1) // 2}"
    )
    distances = calculate_distances(profiles, jobs=jobs, radius=radius)
    matrix = threshold_distances(distances, sigma=sigma, threshold=threshold)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spec.output_path, matrix)
    print(
        f"[{name}] saved {spec.output_path} shape={matrix.shape} "
        f"edges={int(matrix.sum())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sd", "sz", "pems08", "urbanev"],
        choices=["sd", "sz", "pems08", "urbanev"],
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(args.root.resolve())
    for name in args.datasets:
        generate_one(
            name,
            specs[name],
            jobs=args.jobs,
            radius=args.radius,
            sigma=args.sigma,
            threshold=args.threshold,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
