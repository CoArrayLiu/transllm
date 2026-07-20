#!/usr/bin/env python3
"""Build a deterministic, time-stratified evaluation prompt subset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ID_PATTERN = re.compile(
    r"train_(?P<dataset>.+)_region_(?P<start>\d+)_(?P<end>\d+)_len_(?P<window>\d+)"
)


def parse_id(value: str) -> tuple[str, int, int, int]:
    match = ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported prompt id: {value!r}")
    return (
        match.group("dataset"),
        int(match.group("start")),
        int(match.group("end")),
        int(match.group("window")),
    )


def stratified_sample(
    window_indices: list[int],
    sample_count: int,
    seed: int,
) -> list[int]:
    if sample_count <= 0:
        raise ValueError("--windows must be positive")
    if sample_count > len(window_indices):
        raise ValueError(
            f"requested {sample_count} windows but only {len(window_indices)} exist"
        )

    rng = np.random.default_rng(seed)
    bins = np.array_split(np.asarray(sorted(window_indices), dtype=np.int64), sample_count)
    selected = [int(rng.choice(group)) for group in bins]
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prompt", type=Path, required=True)
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-nodes", type=int, default=170)
    args = parser.parse_args()

    with args.input_prompt.open() as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{args.input_prompt} does not contain a non-empty list")

    rows_by_window: dict[int, list[dict]] = defaultdict(list)
    dataset_names: set[str] = set()
    for row in prompts:
        dataset, region_start, region_end, window = parse_id(row["id"])
        if region_end - region_start != 1:
            raise ValueError(f"expected one node per prompt row: {row['id']}")
        dataset_names.add(dataset)
        rows_by_window[window].append(row)

    if len(dataset_names) != 1:
        raise ValueError(f"expected one dataset, found {sorted(dataset_names)}")
    dataset = next(iter(dataset_names))

    node_counts = Counter(len(rows) for rows in rows_by_window.values())
    if node_counts != Counter({args.expected_nodes: len(rows_by_window)}):
        raise ValueError(
            f"not every window contains {args.expected_nodes} nodes: {node_counts}"
        )

    selected_windows = stratified_sample(
        list(rows_by_window),
        sample_count=args.windows,
        seed=args.seed,
    )
    selected_rows = [
        row
        for window in selected_windows
        for row in sorted(
            rows_by_window[window],
            key=lambda item: parse_id(item["id"])[1],
        )
    ]

    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prompt.open("w") as handle:
        json.dump(selected_rows, handle, ensure_ascii=False)

    manifest = {
        "dataset": dataset,
        "seed": args.seed,
        "strategy": "one random window from each equal chronological bin",
        "candidate_window_count": len(rows_by_window),
        "selected_window_count": len(selected_windows),
        "selected_window_indices": selected_windows,
        "nodes_per_window": args.expected_nodes,
        "record_count": len(selected_rows),
        "source_prompt": str(args.input_prompt),
        "output_prompt": str(args.output_prompt),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

