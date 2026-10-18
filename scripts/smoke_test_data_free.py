#!/usr/bin/env python3
"""Read-only integrity and syntax smoke test for the camera-ready release."""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path

FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin", ".pkl", ".joblib"}
REQUIRED = ["README.md", "RELEASE_CHECKLIST.md", "MANIFEST.sha256", "environment.yml",
            "configs/paths.example.yaml", "pipeline/train_balanced_gc_ablation.py",
            "pipeline/eval_dagresidual_ablation.py",
            "scripts/run_camera_ready_oracle.py",
            "scripts/aggregate_camera_ready.py",
            "scripts/prepare_camera_ready_data.py",
            "scripts/validate_data_contract.py",
            "scripts/validate_bth_no_oracle_bundle.py",
            "data/bth_no_oracle_source_8x14/bundle_manifest.json",
            "provenance/main_table_abl_AC_aggregate.json"]

CLI_ENTRYPOINTS = ["pipeline/train_balanced_gc_ablation.py",
                   "pipeline/eval_dagresidual_ablation.py",
                   "scripts/run_camera_ready_oracle.py",
                   "scripts/aggregate_camera_ready.py",
                   "scripts/prepare_camera_ready_data.py",
                   "scripts/validate_data_contract.py"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-cli", action="store_true",
                        help="also launch the three main entry points with --help")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).is_file():
            errors.append(f"missing required file: {rel}")

    manifest = root / "MANIFEST.sha256"
    manifested: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
            if not match:
                errors.append(f"malformed manifest line: {line}")
                continue
            expected, rel = match.groups()
            rel = rel[2:] if rel.startswith("./") else rel
            manifested.add(rel)
            target = root / rel
            if not target.is_file():
                errors.append(f"manifest target missing: {rel}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != expected.lower():
                errors.append(f"checksum mismatch: {rel}")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest
        }
        for rel in sorted(actual - manifested):
            errors.append(f"file missing from manifest: {rel}")
        for rel in sorted(manifested - actual):
            errors.append(f"manifest contains unexpected path: {rel}")

    py_files = sorted(root.rglob("*.py"))
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"Python syntax/read error in {path.relative_to(root)}: {exc}")

    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlink is not allowed in source-only release: {path.relative_to(root)}")
        allowed_prepared_data = (
            path.is_relative_to(root / "data" / "bth_no_oracle_source_8x14")
            and path.suffix.lower() == ".npy"
        )
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES and not allowed_prepared_data:
            errors.append(f"unexpected data/checkpoint artifact: {path.relative_to(root)}")

    if args.check_cli and not errors:
        import subprocess
        for rel in CLI_ENTRYPOINTS:
            result = subprocess.run([sys.executable, str(root / rel), "--help"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                errors.append(f"--help failed for {rel}: {result.stderr[-1000:]}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(py_files)} Python files parse; manifest verified; release checks passed.")
    if args.check_cli:
        print("PASS: primary training/evaluation CLI help checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
