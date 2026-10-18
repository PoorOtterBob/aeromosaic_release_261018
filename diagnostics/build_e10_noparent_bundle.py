#!/usr/bin/env python3
"""Derive an E10 bundle with the parent-label prompt block removed.

The source already carries E10's 14-channel numeric-precipitation tensor.
This script changes only the upstream parent context in prompts.
"""

import argparse
import json
import re
from pathlib import Path


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
    boundaries = (
        prompt.find("Recent precipitation context:", start),
        prompt.find("Predict the final air-quality label", start),
    )
    end = min((pos for pos in boundaries if pos >= 0), default=len(prompt))
    return prompt[:start] + f"Upstream graph context:\n{context}\n" + prompt[end:]


def prompt_text(entry) -> str:
    return str(entry.get("prompt", "")) if isinstance(entry, dict) else str(entry)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source_data_dir)
    out = Path(args.out_dir)
    for split in ("train", "val", "test"):
        source_index = source / "assets" / split / "group_index.jsonl"
        if not source_index.is_file():
            raise FileNotFoundError(source_index)
        dest_assets = out / "assets" / split
        dest_assets.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in source_index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompts = json.loads(Path(row["prompts_path"]).read_text(encoding="utf-8"))
            rewritten = []
            for city, entry in enumerate(prompts):
                context = "unknown/no-parent"
                rewritten.append({"prompt": replace_upstream_block(prompt_text(entry), context)})
            prompt_path = dest_assets / f"{row['group_id']}_e10_noparent_prompts.json"
            prompt_path.write_text(json.dumps(rewritten, ensure_ascii=False), encoding="utf-8")
            derived = dict(row)
            derived["prompts_path"] = str(prompt_path)
            rows.append(derived)
        with (dest_assets / "group_index.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "raw_ts": "Preserved from E10: [8,14] including numeric precipitation channel.",
                "parent_context": "Removed only target-date parent-label context.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
