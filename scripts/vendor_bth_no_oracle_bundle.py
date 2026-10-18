#!/usr/bin/env python3
"""Create a self-contained portable copy of the historical BTH 8x14 source bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
FIELDS = {
    "raw_ts_path": "raw_ts.npy",
    "labels_path": "labels.npy",
    "dag_adj_path": "dag_adj.npy",
    "dag_level_path": "dag_level.npy",
    "is_root_path": "is_root.npy",
    "city_names_path": "city_names.json",
    "prompts_path": "prompts.json",
}


def resolve(raw: str, source: Path, project_root: Path) -> Path:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [project_root / path, source / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve {raw!r}; tried {candidates}")


def portable(path: Path, release_root: Path) -> str:
    try:
        return path.resolve().relative_to(release_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("--out-dir must be inside --release-root") from exc


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="root used by relative paths in the historical source index")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True,
                        help="paths in the new index are stored relative to this directory")
    args = parser.parse_args()

    source = args.source_data_dir.resolve()
    project_root = args.project_root.resolve()
    out = args.out_dir.resolve()
    release_root = args.release_root.resolve()
    portable(out, release_root)
    out.mkdir(parents=True, exist_ok=False)

    counts: dict[str, int] = {}
    for split in SPLITS:
        source_index = source / "assets" / split / "group_index.jsonl"
        if not source_index.is_file():
            raise FileNotFoundError(source_index)
        output_index = out / "assets" / split / "group_index.jsonl"
        output_index.parent.mkdir(parents=True, exist_ok=True)
        output_rows = []
        for line in source_index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row["group_id"])
            group_dir = out / "assets" / split / group_id
            group_dir.mkdir(parents=True, exist_ok=False)
            output_row = {
                key: value for key, value in row.items()
                if not key.endswith("_path")
            }
            for field, filename in FIELDS.items():
                if field not in row:
                    raise ValueError(f"{split}/{group_id}: missing {field}")
                src = resolve(str(row[field]), source, project_root)
                dst = group_dir / filename
                shutil.copy2(src, dst)
                output_row[field] = portable(dst, release_root)

            raw = np.load(group_dir / "raw_ts.npy", mmap_mode="r")
            labels = np.load(group_dir / "labels.npy", mmap_mode="r")
            adjacency = np.load(group_dir / "dag_adj.npy", mmap_mode="r")
            if raw.shape != (13, 8, 14):
                raise ValueError(f"{split}/{group_id}: raw_ts shape {raw.shape}, expected (13,8,14)")
            if labels.shape != (13,) or adjacency.shape != (13, 13):
                raise ValueError(
                    f"{split}/{group_id}: labels/adjacency shapes {labels.shape}/{adjacency.shape}"
                )
            if not np.isfinite(raw).all() or not np.isfinite(adjacency).all():
                raise ValueError(f"{split}/{group_id}: non-finite numerical input")
            if not np.issubdtype(labels.dtype, np.integer) or not np.all((labels >= 0) & (labels <= 5)):
                raise ValueError(f"{split}/{group_id}: invalid class labels")
            prompts = json.loads((group_dir / "prompts.json").read_text(encoding="utf-8"))
            names = json.loads((group_dir / "city_names.json").read_text(encoding="utf-8"))
            if len(prompts) != 13 or len(names) != 13:
                raise ValueError(f"{split}/{group_id}: expected 13 prompts and city names")

            parent_count_path = group_dir / "parent_count.npy"
            np.save(parent_count_path, (adjacency > 0).sum(axis=1).astype(np.int64))
            output_row["parent_count_path"] = portable(parent_count_path, release_root)
            output_rows.append(output_row)

        with output_index.open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(output_rows)

    manifest = {
        "asset": "AeroMosaic historical BTH numerical-precipitation source bundle",
        "purpose": "input for the matched no-oracle parent-information diagnostics",
        "raw_ts_shape": [13, 8, 14],
        "splits": counts,
        "source_bundle": "preserved BTH numerical-precipitation experiment asset",
        "source_index_sha256": {
            split: digest(source / "assets" / split / "group_index.jsonl") for split in SPLITS
        },
        "portability": "all indexed files are contained under this bundle; index paths are release-root-relative",
        "parent_count": "derived as the number of positive entries per stored adjacency row",
        "provenance_boundary": (
            "This vendors the preserved experiment-time prepared input. It does not recreate the "
            "14-channel tensor from public raw KnowAir data."
        ),
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote {out} with {counts}")


if __name__ == "__main__":
    main()
