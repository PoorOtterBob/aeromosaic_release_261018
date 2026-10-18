#!/usr/bin/env python3
"""Run the exact six-region, three-seed abl_AC main-table protocol.

This entry point intentionally consumes the frozen experiment-time bundles. It
does not rebuild their DAGs with the later unified reconstruction utility.
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml


REGIONS = ("BTH", "YRD", "SDP", "ZHD", "CJD", "CYB")
EXPECTED_SEEDS = (42, 123, 456)
ABLATION_MASK = "0,11,12"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required = ("python_bin", "project_root", "model_dir", "output_root", "seeds", "regions")
    missing = [key for key in required if not cfg or key not in cfg]
    if missing:
        parser.error(f"config missing keys: {missing}")
    if cfg.get("protocol") not in (None, "main_table_abl_AC"):
        parser.error("protocol must be main_table_abl_AC")
    data_map = cfg["regions"]
    if set(data_map) != set(REGIONS):
        parser.error(f"regions must define exactly {REGIONS}")
    seeds = tuple(int(seed) for seed in cfg["seeds"])
    if seeds != EXPECTED_SEEDS:
        parser.error(f"main_table_abl_AC seeds must be exactly {EXPECTED_SEEDS}")

    root = Path(cfg["project_root"]).expanduser().resolve()
    python = str(Path(cfg["python_bin"]).expanduser())
    model = Path(cfg["model_dir"]).expanduser().resolve()
    output = Path(cfg["output_root"]).expanduser().resolve() / "abl_AC"
    gpu = str(cfg.get("gpu", "0"))
    trainer = root / "pipeline" / "train_balanced_gc_ablation.py"
    evaluator = root / "pipeline" / "eval_dagresidual_ablation.py"
    validator = root / "scripts" / "validate_data_contract.py"
    for entrypoint in (trainer, evaluator, validator):
        if not entrypoint.is_file():
            parser.error(f"entry point not found: {entrypoint}")
    if not args.dry_run and (not Path(python).is_file() or not model.is_dir()):
        parser.error("python_bin and model_dir must exist; use --dry-run to inspect commands")

    # Exact flags from the archived 2026-04-16 abl_AC launcher. Defaults not
    # present in that launcher are deliberately not overridden here.
    common_train = [
        "--train_split", "train", "--eval_split", "val", "--bf16",
        "--root_aware_loss", "--root_label_extra", "3:2.0,4:3.0",
        "--max_root_sample_weight", "4.0", "--root_region_env_fusion",
        "--root_region_exclude_self", "--num_train_epochs", "3",
        "--learning_rate", "1e-4", "--gradient_accumulation_steps", "2",
        "--logging_steps", "10", "--auto_num_labels", "--root_weight", "1.5",
        "--root_region_env_weight", "0.8", "--dag_residual_enable",
        "--dag_residual_alpha_init", "-3.0", "--dag_residual_hidden_dim", "256",
        "--ablation_mask", ABLATION_MASK,
    ]

    for region in REGIONS:
        data = Path(data_map[region]).expanduser().resolve()
        if not args.dry_run and not data.is_dir():
            parser.error(f"frozen main-table bundle for {region} does not exist: {data}")
        if not args.dry_run:
            # Generic contract only: historical bundles do not claim the later
            # binary/last-historical reconstruction manifest.
            subprocess.run(
                [python, str(validator), "--bundle", str(data),
                 "--region", region, "--project-root", str(root)],
                cwd=root, check=True,
            )

        for seed in seeds:
            run_dir = output / f"{region}_seed{seed}"
            eval_dir = output / f"{region}_seed{seed}_eval_test"
            train = [
                python, str(trainer), "--model_name_or_path", str(model),
                "--data_dir", str(data), "--out_dir", str(run_dir),
                *common_train, "--seed", str(seed),
            ]
            evaluate = [
                python, str(evaluator), "--model_name_or_path", str(model),
                "--adapter_dir", str(run_dir / "final_adapter"),
                "--data_dir", str(data), "--out_dir", str(eval_dir),
                "--eval_splits", "test", "--bf16", "--trust_remote_code",
                "--ablation_mask", ABLATION_MASK,
            ]
            if args.dry_run:
                print("TRAIN", " ".join(train))
                print("EVAL ", " ".join(evaluate))
                continue

            adapter = run_dir / "final_adapter"
            if not adapter.is_dir():
                run_dir.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                    subprocess.run(train, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            if not adapter.is_dir():
                raise RuntimeError(f"training finished without final_adapter: {adapter}")

            summary = eval_dir / "test" / "summary.json"
            if not summary.is_file():
                eval_dir.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                with (eval_dir / "eval.log").open("w", encoding="utf-8") as log:
                    subprocess.run(evaluate, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            print(f"Completed {region} seed={seed}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
