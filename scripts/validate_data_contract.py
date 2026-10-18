#!/usr/bin/env python3
"""Validate a KnowAir group-bundle directory without exposing or copying its data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = ("raw_ts_path", "labels_path", "dag_adj_path", "dag_level_path",
                   "is_root_path", "parent_count_path", "city_names_path", "prompts_path")
SPLITS = ("train", "val", "test")
REGION_SIZES = {"BTH": 13, "YRD": 16, "SDP": 15, "ZHD": 17, "CJD": 21, "CYB": 17}


def resolve(bundle: Path, raw: str, project_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = ((bundle / path).resolve(), (project_root / path).resolve())
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--region", required=True, choices=("BTH", "YRD", "SDP", "ZHD", "CJD", "CYB"))
    parser.add_argument("--project-root", type=Path,
                        help="base for paths stored relative to the experiment project; defaults to the bundle")
    parser.add_argument("--paper-protocol", action="store_true",
                        help="require the camera-ready binary, last-historical-wind bundle contract")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    project_root = args.project_root.resolve() if args.project_root else bundle
    errors: list[str] = []
    groups = 0
    if args.paper_protocol:
        manifest_path = bundle / "dataset_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"paper-protocol manifest missing: {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("region") != args.region:
                errors.append("manifest region does not match requested region")
            if manifest.get("dag_wind") != "last observation in historical window, channels 7-8":
                errors.append("manifest does not certify last-historical wind")
            if manifest.get("dag", {}).get("adjacency") != "binary destination-by-source":
                errors.append("manifest does not certify binary destination-by-source adjacency")
            if not manifest.get("precipitation_text", "").startswith("raw metres"):
                errors.append("manifest does not certify original-metre precipitation text")

    for split in SPLITS:
        index = bundle / "assets" / split / "group_index.jsonl"
        if not index.is_file():
            errors.append(f"missing split index: {index}")
            continue
        for line_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            groups += 1
            try:
                row = json.loads(line)
                missing = [key for key in REQUIRED_FIELDS if key not in row]
                if missing:
                    raise ValueError(f"missing fields {missing}")
                paths = {key: resolve(bundle, str(row[key]), project_root) for key in REQUIRED_FIELDS}
                for key, path in paths.items():
                    if not path.is_file():
                        raise FileNotFoundError(f"{key}: {path}")
                raw = np.load(paths["raw_ts_path"], mmap_mode="r")
                labels = np.load(paths["labels_path"], mmap_mode="r")
                adj = np.load(paths["dag_adj_path"], mmap_mode="r")
                level = np.load(paths["dag_level_path"], mmap_mode="r")
                root = np.load(paths["is_root_path"], mmap_mode="r")
                parent_count = np.load(paths["parent_count_path"], mmap_mode="r")
                n = raw.shape[0] if raw.ndim else -1
                if raw.shape != (n, 8, 13):
                    raise ValueError(f"raw_ts expected (N,8,13), got {raw.shape}")
                if n != REGION_SIZES[args.region]:
                    raise ValueError(
                        f"{args.region} expected {REGION_SIZES[args.region]} nodes, got {n}"
                    )
                if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
                    raise ValueError("raw_ts must be numeric and contain only finite values")
                if labels.shape != (n,):
                    raise ValueError(f"labels expected ({n},), got {labels.shape}")
                if not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError(f"labels must use an integer dtype, got {labels.dtype}")
                if not np.all((labels >= 0) & (labels <= 5)):
                    bad = np.unique(labels[(labels < 0) | (labels > 5)]).tolist()
                    raise ValueError(f"labels must be class IDs 0 through 5, got {bad}")
                if adj.shape != (n, n):
                    raise ValueError(f"dag_adj expected ({n},{n}), got {adj.shape}")
                if not np.issubdtype(adj.dtype, np.number) or not np.isfinite(adj).all():
                    raise ValueError("dag_adj must be numeric and contain only finite values")
                if np.any(adj < 0):
                    raise ValueError("dag_adj must contain non-negative edge weights")
                if np.any(np.diag(adj) != 0):
                    raise ValueError("dag_adj must not contain self-loops")
                if level.shape != (n,) or root.shape != (n,) or parent_count.shape != (n,):
                    raise ValueError(
                        f"dag_level/is_root/parent_count expected ({n},), got "
                        f"{level.shape}/{root.shape}/{parent_count.shape}"
                    )
                for field_name, array in (("dag_level", level), ("is_root", root),
                                          ("parent_count", parent_count)):
                    if not np.issubdtype(array.dtype, np.integer):
                        raise ValueError(f"{field_name} must use an integer dtype, got {array.dtype}")
                if np.any(level < 0):
                    raise ValueError("dag_level must be non-negative")
                if not np.all((root == 0) | (root == 1)):
                    raise ValueError("is_root must contain only 0/1 values")
                if np.any(parent_count < 0):
                    raise ValueError("parent_count must be non-negative")
                expected_parent_count = (adj > 0).sum(axis=1).astype(parent_count.dtype)
                if not np.array_equal(parent_count, expected_parent_count):
                    raise ValueError("parent_count does not match positive incoming adjacency entries")
                names = json.loads(paths["city_names_path"].read_text(encoding="utf-8"))
                if not isinstance(names, list) or len(names) != n:
                    raise ValueError("bundle must have one city name per node")
                if not all(isinstance(name, str) and name.strip() for name in names):
                    raise ValueError("city_names must contain non-empty strings")
                if len(set(names)) != n:
                    raise ValueError("city_names must be unique within a region")
                if args.paper_protocol:
                    if not np.all((adj == 0) | (adj == 1)):
                        raise ValueError("paper DAG adjacency must be binary")
                    if not np.array_equal(root, (adj.sum(axis=1) == 0).astype(root.dtype)):
                        raise ValueError("root mask does not match zero in-degree rows")
                    dst, src = np.nonzero(adj)
                    if not np.all(level[dst] > level[src]):
                        raise ValueError("DAG levels must increase along source-to-destination edges")
                prompts = json.loads(paths["prompts_path"].read_text(encoding="utf-8"))
                if not isinstance(prompts, list) or len(prompts) != n:
                    raise ValueError(f"prompts expected list length {n}")
                if prompts and not (isinstance(prompts[0], str) or
                                    isinstance(prompts[0], dict) and "prompt" in prompts[0]):
                    raise ValueError("prompt entries must be strings or dictionaries with a prompt field")
                if args.paper_protocol:
                    for entry in prompts:
                        prompt = entry if isinstance(entry, str) else entry["prompt"]
                        for phrase in ("oracle protocol: target-date parent labels",
                                       "Recent precipitation context:",
                                       "Output only one label token: 0, 1, 2, 3, 4, or 5."):
                            if phrase not in prompt:
                                raise ValueError(f"paper prompt missing {phrase!r}")
                    for i, entry in enumerate(prompts):
                        prompt = entry if isinstance(entry, str) else entry["prompt"]
                        parents = np.flatnonzero(adj[i]).tolist()
                        if not parents:
                            if "Upstream: none (root)." not in prompt:
                                raise ValueError(f"root city {i} lacks the no-parent marker")
                        else:
                            for j in parents:
                                field = f"- {names[j]}: label={int(labels[j])}"
                                if field not in prompt:
                                    raise ValueError(f"city {i} missing oracle parent field {field!r}")
            except Exception as exc:
                errors.append(f"{split}/group_index.jsonl:{line_no}: {exc}")

    if errors:
        print(f"FAIL: region={args.region}, groups_seen={groups}")
        for error in errors[:100]:
            print(f"- {error}")
        return 1
    print(f"PASS: region={args.region}, bundle={bundle}, groups={groups}, splits={','.join(SPLITS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
