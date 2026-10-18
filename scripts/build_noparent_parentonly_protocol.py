#!/usr/bin/env python3
"""Build a NoParent protocol that removes only oracle parent labels."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

PARENT_BLOCK = re.compile(r"Upstream parent-label context:.*?\. Predict the PM2\.5 risk class\.")
REPLACEMENT = "Upstream parent-label context: unknown/no-parent. Predict the PM2.5 risk class."

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

for split in ("train", "val", "test"):
    source_index = args.source / "assets" / split / "group_index.jsonl"
    target_index = args.output / "assets" / split / "group_index.jsonl"
    target_index.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source_index.read_text(encoding="utf-8").splitlines() if line]
    with target_index.open("w", encoding="utf-8") as stream:
        for row in rows:
            prompts = json.loads(Path(row["prompts_path"]).read_text(encoding="utf-8"))
            rewritten = []
            for entry in prompts:
                prompt = entry["prompt"] if isinstance(entry, dict) else str(entry)
                updated, count = PARENT_BLOCK.subn(REPLACEMENT, prompt)
                if count != 1:
                    raise ValueError(f"{split}/{row['group_id']}: expected one parent block, got {count}")
                if "DAG level:" not in updated or "Recent precipitation:" not in updated:
                    raise ValueError(f"{split}/{row['group_id']}: non-parent context was lost")
                if isinstance(entry, dict):
                    rewritten.append({**entry, "prompt": updated})
                else:
                    rewritten.append(updated)
            group_dir = target_index.parent / row["group_id"]
            group_dir.mkdir(exist_ok=True)
            prompt_path = group_dir / "prompts.json"
            prompt_path.write_text(json.dumps(rewritten), encoding="utf-8")
            row["prompts_path"] = str(prompt_path)
            stream.write(json.dumps(row) + "\n")
    print(f"{split}: {len(rows)} groups")
