#!/usr/bin/env python3
"""Materialize the BTH group-index splits in the formal region-classifier NPZ format."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source, out = Path(args.source_data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (source / "assets" / split / "group_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        x, y, dates, dags, group_ids = [], [], [], [], []
        city_indices = None
        for index, row in enumerate(rows):
            raw = np.load(row["raw_ts_path"]).astype(np.float32)
            labels = np.load(row["labels_path"]).astype(np.int64)
            dag_adj = np.load(row["dag_adj_path"]).astype(np.float32)
            if raw.ndim != 3 or raw.shape[1:] != (8, 13):
                raise ValueError(f"Unexpected raw_ts shape {raw.shape} for {row['group_id']}")
            if labels.shape != (raw.shape[0],):
                raise ValueError(f"Unexpected labels shape {labels.shape} for {row['group_id']}")
            x.append(np.transpose(raw, (1, 0, 2)))
            y.append(labels)
            dags.append(dag_adj)
            group_ids.append(str(row["group_id"]))
            dates.append(row.get("target_date", row["group_id"]))
            city_indices = np.arange(raw.shape[0], dtype=np.int64)
        years = np.asarray([int(str(day)[:4]) if str(day)[:4].isdigit() else -1 for day in dates], dtype=np.int64)
        np.savez_compressed(
            out / f"{split}.npz",
            x=np.stack(x).astype(np.float32),
            y=np.stack(y).astype(np.int64),
            target_day_index=np.arange(len(rows), dtype=np.int64),
            target_year=years,
            target_dates=np.asarray(dates),
            region_city_indices=city_indices,
            feature_names=np.asarray([f"raw_feature_{i}" for i in range(13)]),
            dag_adj=np.stack(dags).astype(np.float32),
            group_ids=np.asarray(group_ids),
        )


if __name__ == "__main__":
    main()
