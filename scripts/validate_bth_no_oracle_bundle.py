#!/usr/bin/env python3
"""Validate the vendored, self-contained BTH 8x14 no-oracle source bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SPLITS = {"train": 730, "val": 365, "test": 365}
PATH_FIELDS = (
    "raw_ts_path",
    "labels_path",
    "dag_adj_path",
    "dag_level_path",
    "is_root_path",
    "parent_count_path",
    "city_names_path",
    "prompts_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default="data/bth_no_oracle_source_8x14",
        help="Bundle directory, relative to --release-root unless absolute.",
    )
    parser.add_argument(
        "--release-root",
        default=".",
        help="Release root used to resolve paths stored in JSONL indexes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.release_root).resolve()
    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = (root / bundle).resolve()
    if not bundle.is_dir():
        raise SystemExit(f"bundle directory not found: {bundle}")

    seen: set[str] = set()
    total = 0
    for split, expected_count in SPLITS.items():
        index = bundle / "assets" / split / "group_index.jsonl"
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected_count:
            raise AssertionError(f"{split}: expected {expected_count} rows, found {len(rows)}")
        for row in rows:
            group_id = str(row["group_id"])
            if group_id in seen:
                raise AssertionError(f"duplicate group_id: {group_id}")
            seen.add(group_id)
            resolved: dict[str, Path] = {}
            for field in PATH_FIELDS:
                path = Path(row[field])
                path = path.resolve() if path.is_absolute() else (root / path).resolve()
                if not path.is_file():
                    raise AssertionError(f"{group_id}: missing {field}: {path}")
                if bundle not in path.parents:
                    raise AssertionError(f"{group_id}: {field} escapes bundle: {path}")
                resolved[field] = path

            raw = np.load(resolved["raw_ts_path"], allow_pickle=False)
            labels = np.load(resolved["labels_path"], allow_pickle=False)
            adj = np.load(resolved["dag_adj_path"], allow_pickle=False)
            levels = np.load(resolved["dag_level_path"], allow_pickle=False)
            roots = np.load(resolved["is_root_path"], allow_pickle=False)
            parents = np.load(resolved["parent_count_path"], allow_pickle=False)
            if raw.shape != (13, 8, 14):
                raise AssertionError(f"{group_id}: raw_ts shape {raw.shape}, expected (13, 8, 14)")
            if labels.shape != (13,) or adj.shape != (13, 13):
                raise AssertionError(f"{group_id}: invalid labels/adjacency shapes")
            if levels.shape != (13,) or roots.shape != (13,) or parents.shape != (13,):
                raise AssertionError(f"{group_id}: invalid DAG metadata shapes")
            if not np.isfinite(raw).all() or not np.isfinite(adj).all():
                raise AssertionError(f"{group_id}: non-finite numeric values")
            if labels.min() < 0 or labels.max() > 5:
                raise AssertionError(f"{group_id}: labels outside 0..5")
            expected_parents = (adj > 0).sum(axis=1).astype(parents.dtype, copy=False)
            if not np.array_equal(parents, expected_parents):
                raise AssertionError(f"{group_id}: parent_count does not match adjacency")
            names = json.loads(resolved["city_names_path"].read_text(encoding="utf-8"))
            prompts = json.loads(resolved["prompts_path"].read_text(encoding="utf-8"))
            if len(names) != 13 or len(prompts) != 13:
                raise AssertionError(f"{group_id}: expected 13 city names and prompts")
            total += 1
        print(f"PASS {split}: {len(rows)} groups")

    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    manifest_total = sum(int(value) for value in manifest.get("splits", {}).values())
    if manifest_total != total:
        raise AssertionError("bundle_manifest split counts mismatch")
    print(f"PASS portable BTH no-oracle bundle: {total} groups; all paths are self-contained")


if __name__ == "__main__":
    main()
