#!/usr/bin/env python3
"""Build E7 prompts containing only observed parent PM2.5 from the prior day."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PARENT_LABEL_BLOCK = re.compile(
    r"Upstream parent-label context:.*?\. Predict the PM2\.5 risk class\.",
    flags=re.DOTALL,
)


def replace_upstream_block(prompt: str, context: str) -> str:
    replacement = (
        f"Upstream parent-label context: {context}. "
        "Predict the PM2.5 risk class."
    )
    rewritten, count = PARENT_LABEL_BLOCK.subn(replacement, prompt, count=1)
    if count == 1:
        return rewritten

    # Compatibility with the older multiline prompt schema.
    start = prompt.find("Upstream graph context:")
    if start < 0:
        raise ValueError("source prompt has no upstream graph context block")
    boundaries = [
        prompt.find("Recent precipitation context:", start),
        prompt.find("Predict the final air-quality label", start),
    ]
    end = min((x for x in boundaries if x >= 0), default=len(prompt))
    return prompt[:start] + f"Upstream graph context:\n{context}\n" + prompt[end:]


def load_json(path: str):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def build_split(source: Path, destination: Path, split: str) -> None:
    rows = []
    out_assets = destination / "assets" / split
    out_assets.mkdir(parents=True, exist_ok=True)
    index_path = source / "assets" / split / "group_index.jsonl"

    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cities = [str(x) for x in load_json(row["city_names_path"])]
        prompts = load_json(row["prompts_path"])
        raw_ts = np.load(row["raw_ts_path"]).astype(np.float32)
        dag_adj = np.load(row["dag_adj_path"]).astype(np.float32)
        is_root = np.load(row["is_root_path"]).astype(np.int64)
        previous_day_pm25 = raw_ts[:, -1, 0]

        rewritten = []
        for child, entry in enumerate(prompts):
            prompt = str(entry.get("prompt", "")) if isinstance(entry, dict) else str(entry)
            parents = np.flatnonzero(dag_adj[child] > 0).tolist()
            if int(is_root[child]) == 1 or not parents:
                context = "unknown/no-parent"
            else:
                context = "; ".join(
                    f"{cities[parent]}: previous-day observed PM2.5={previous_day_pm25[parent]:.1f} ug/m3"
                    for parent in parents
                )
            rewritten.append({"prompt": replace_upstream_block(prompt, context)})

        prompt_path = out_assets / f"{row['group_id']}_lag1_pm25_prompts.json"
        prompt_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")
        output_row = dict(row)
        output_row["prompts_path"] = str(prompt_path)
        rows.append(output_row)

    with (out_assets / "group_index.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    source, out = Path(args.source_data_dir), Path(args.out_dir)
    for split in ("train", "val", "test"):
        build_split(source, out, split)
    (out / "manifest.json").write_text(json.dumps({
        "protocol": "E7 lag-1 observed parent PM2.5 context",
        "parent_context": "Each parent city's own raw_ts[-1, 0] from the input window.",
        "target_day_label_access": False,
        "oracle_parent_label_access": False,
        "online_prediction_access": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
