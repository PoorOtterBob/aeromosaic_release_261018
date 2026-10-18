#!/usr/bin/env python3
"""Migrate an older bundle by adding parent_count.npy/path to every group."""
import argparse
import json
from pathlib import Path

import numpy as np


def resolve(bundle: Path, raw: str, project_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = ((bundle / path).resolve(), (project_root / path).resolve())
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="regional bundle directory")
    parser.add_argument("--project-root", type=Path,
                        help="base for project-relative indexed paths; defaults to bundle root")
    args = parser.parse_args()
    bundle = args.root.resolve()
    project_root = args.project_root.resolve() if args.project_root else bundle
    for split in ("train", "val", "test"):
        index = bundle / "assets" / split / "group_index.jsonl"
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        for row in rows:
            dag_path = resolve(bundle, str(row["dag_adj_path"]), project_root)
            group = resolve(bundle, str(row["prompts_path"]), project_root).parent
            output = group / "parent_count.npy"
            adjacency = np.load(dag_path, allow_pickle=False)
            np.save(output, (adjacency > 0).sum(axis=1).astype(np.int64))
            try:
                row["parent_count_path"] = output.resolve().relative_to(project_root).as_posix()
            except ValueError:
                row["parent_count_path"] = str(output.resolve())
        index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
        print(f"{split}: {len(rows)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
