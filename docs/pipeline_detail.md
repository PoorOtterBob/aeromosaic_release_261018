# Current AeroMosaic pipeline map

This file replaces an obsolete internal note that referred to pre-release
entry points, five-seed experiments and scripts not included in this package.
Use the following files as the executable authority:

1. `scripts/prepare_camera_ready_data.py` builds runnable regional bundles from
   CauAir's processed KnowAir `his.npz`, original precipitation observations
   and `city.txt`.
2. `scripts/validate_data_contract.py` checks every field required by the
   trainer and evaluator, including city names and parent counts.
3. `scripts/run_camera_ready_oracle.py` launches the six-region, three-seed
   `abl_AC` protocol using `configs/paths.yaml`.
4. `pipeline/train_balanced_gc_ablation.py` is the authoritative trainer.
5. `pipeline/eval_dagresidual_ablation.py` is the authoritative evaluator.
6. `scripts/aggregate_camera_ready.py` aggregates the 18 test summaries.

The complete commands, output layout and the distinction between newly built
software inputs and historical main-table DAG assets are documented in
`docs/reproduction.md`. Tensor and JSONL requirements are in
`docs/data_contract.md`; paper-to-code evidence is in
`docs/paper_code_alignment.md`.

The dated `*_20260417.md` files are historical records only and are not current
execution guides.
