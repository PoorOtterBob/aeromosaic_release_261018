# Main-table provenance

This directory records the archived experiment chain used to identify the
paper's six-region main-table results. It contains no model weights, processed
inputs, predictions or private station data.

- `archived_run_ablation_v2_20260416.sh` is the preserved launcher. It is kept
  for provenance only because its absolute paths are machine-specific. SHA-256:
  `8f0dbcd51b1b32ed5ba1185d2b733f06348b0ea32446702feae89bbc4f1730a7`.
- `main_table_abl_AC_aggregate.json` was produced during release preparation by
  running the aggregator over 18 archived `abl_AC` test summaries stored
  outside this source package. It records all
  per-seed Micro-F1/Macro-F1 values, population standard deviations, 12
  successful four-decimal paper-mean checks and 12 successful population-SD
  checks. The audit corrected three stale manuscript error bars without
  changing any mean. The individual historical summaries are ordinary
  evaluator outputs, are not needed to run the software, and are not bundled.
- `pipeline/train_balanced_gc_ablation.py` SHA-256:
  `8277fac6a49d81a3ec8cc3234294e56e002bd154c38da89e9a2aee78f69867e3`.
- `pipeline/eval_dagresidual_ablation.py` SHA-256:
  `c4d240d131b14bf4d2f4f3562e810f7e43dfe990776f9786759a58cff4bdda9d`.

The executable portable entry point is `scripts/run_camera_ready_oracle.py`.
It reproduces the launcher's `abl_AC` settings while taking external paths from
`configs/paths.yaml`.
