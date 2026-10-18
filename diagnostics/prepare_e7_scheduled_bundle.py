#!/usr/bin/env python3
"""Create deterministic scheduled-sampling prompts for E7.

The source oracle bundle supplies the prompt template and labels. A separate
no-parent warm-up model supplies rollout predictions on train/val groups. For
each parent slot, the generated training prompt uses its prediction with
probability p and its ground-truth class otherwise. Roots always remain
unknown/no-parent. Test is deliberately not generated here: it is evaluated
by the strict online rollout evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

CONTEXT_RE = re.compile(r"Upstream parent-label context: .*? Predict the PM2\.5 risk class\.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_data_dir", required=True)
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_pred_prob", type=float, default=0.75)
    parser.add_argument("--val_pred_prob", type=float, default=1.0)
    return parser.parse_args()


def replace_context(prompt: str, context: str) -> str:
    start = prompt.find("Upstream graph context:")
    if start >= 0:
        boundaries = [
            prompt.find("Recent precipitation context:", start),
            prompt.find("Predict the final air-quality label", start),
        ]
        end = min((x for x in boundaries if x >= 0), default=len(prompt))
        return prompt[:start] + f"Upstream graph context:\n{context}\n" + prompt[end:]
    repl = f"Upstream parent-label context: {context} Predict the PM2.5 risk class."
    if "Upstream parent-label context:" in prompt:
        return CONTEXT_RE.sub(repl, prompt)
    return prompt + " " + repl


def read_predictions(path: Path):
    pred = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pred[(str(row["group_id"]), str(row["city_name"]))] = int(row["pred"])
    return pred


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def generate_split(args, split: str, pred_prob: float, rng: random.Random):
    source_index = Path(args.oracle_data_dir) / "assets" / split / "group_index.jsonl"
    output_assets = Path(args.out_dir) / "assets" / split
    output_assets.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(Path(args.rollout_dir) / split / "predictions.csv")
    out_rows = []

    for line in source_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        group_id = str(row["group_id"])
        city_names = [str(x) for x in load_json(Path(row["city_names_path"]))]
        prompt_entries = load_json(Path(row["prompts_path"]))
        prompts = [str(x.get("prompt", "")) if isinstance(x, dict) else str(x) for x in prompt_entries]
        labels = np.load(row["labels_path"]).astype(np.int64)
        dag_adj = np.load(row["dag_adj_path"]).astype(np.float32)
        is_root = np.load(row["is_root_path"]).astype(np.int64)

        generated = []
        for child, prompt in enumerate(prompts):
            parents = np.flatnonzero(dag_adj[child] > 0).tolist()
            if int(is_root[child]) == 1 or not parents:
                context = "unknown/no-parent"
            else:
                parts = []
                for parent in parents:
                    rollout_label = predictions[(group_id, city_names[parent])]
                    label = rollout_label if rng.random() < pred_prob else int(labels[parent])
                    parts.append(f"{city_names[parent]}: class {label}")
                context = ", ".join(parts)
            generated.append({"prompt": replace_context(prompt, context)})

        prompt_path = output_assets / f"{group_id}_scheduled_prompts.json"
        prompt_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        out_row = dict(row)
        out_row["prompts_path"] = str(prompt_path)
        out_rows.append(out_row)

    with (output_assets / "group_index.jsonl").open("w", encoding="utf-8") as stream:
        for row in out_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not 0.0 <= args.train_pred_prob <= 1.0 or not 0.0 <= args.val_pred_prob <= 1.0:
        raise ValueError("prediction probabilities must be in [0, 1]")
    rng = random.Random(args.seed)
    generate_split(args, "train", args.train_pred_prob, rng)
    generate_split(args, "val", args.val_pred_prob, rng)
    manifest = {
        "protocol": "E7 scheduled sampling",
        "seed": args.seed,
        "train_pred_prob": args.train_pred_prob,
        "val_pred_prob": args.val_pred_prob,
        "test_parent_source": "strict online self-predicted rollout; no target-date true parent labels",
    }
    (Path(args.out_dir) / "scheduled_sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
