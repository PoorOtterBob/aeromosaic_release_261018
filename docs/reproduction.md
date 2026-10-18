# Camera-ready reproduction recipes

These commands cover AeroMosaic's primary six-region oracle-parent experiment and the available BTH no-oracle controls. The six frozen main-table bundles, model weights, and third-party baseline code are external; the prepared BTH no-oracle source bundle is included.

## 1. Environment and local paths

Use Linux with Python 3.11, CUDA 12.4, and a compatible GPU. From the repository root:

```bash
conda env create -f environment.yml
conda activate aeromosaic-repro
cp configs/paths.example.yaml configs/paths.yaml
```

Edit `configs/paths.yaml` locally. Set `project_root` to this checkout, `python_bin` to the environment's Python executable, and `model_dir` to ChatTS-8B-1103 downloaded from the official model page at revision `7216a84034063c2d394901fe45d20a1e8d773a1f`. The model checkpoint is large and is not included. Obtain it only after reviewing the model-card terms.

For an exact archived main-table rerun, each `regions` entry points to the corresponding experiment-time `*_dag_group_bundle_precip*` directory named in `configs/paths.example.yaml`. For a packaged-workflow or prospective run, the same fields may point to bundles generated below; such runs exercise the complete software interface but must not be presented as regenerating the historical main-table DAGs. The YAML `seeds` field is executable configuration and must remain `[42, 123, 456]` for `main_table_abl_AC`.

### Historical input-data pointer

The paper's historical input is the original KnowAir benchmark used by PM2.5-GNN, not the later KnowAir-V2 release. CauAir publicly provides the processed KnowAir [`data_knowair/24_24/all/his.npz`](https://github.com/PoorOtterBob/CauAir/tree/main/data_knowair/24_24/all) expected by this builder. The [PM2.5-GNN Dataset instructions](https://github.com/shuowang-ai/PM2.5-GNN#dataset) link the original KnowAir archive on [Zenodo (DOI 10.5281/zenodo.23339627)](https://doi.org/10.5281/zenodo.23339627), and list Google Drive and Baidu Cloud mirrors (`KnowAir.npy`; the README lists Baidu access code `t82d`). The same project provides [`city.txt`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/data/city.txt). The builder uses `his.npz` for the normalized 13-channel tensor, `KnowAir.npy` column 12 for precipitation in metres, and `city.txt` for city order and coordinates.

The raw KnowAir file alone is not an AeroMosaic-ready bundle. In this release, a bundle means the processed regional directory consumed by the trainer/evaluator. `scripts/prepare_camera_ready_data.py` takes CauAir's normalized `his.npz` (`data`, `mean`, `std`; `data` shape `(11688,184,13)`), upstream-format `city.txt`, and aligned raw precipitation in metres, then creates all required arrays, parent counts, city names, prompts and split indices. It provides a closed packaged-data workflow, but its unified DAG rule does not recreate the five-region historical main-table graphs. SDP contains 15 cities and does not include Dezhou.

For prospective data preparation or reconstruction diagnostics only, run:

```bash
git clone https://github.com/PoorOtterBob/CauAir.git /path/to/CauAir

python scripts/prepare_camera_ready_data.py \
  --his /path/to/CauAir/data_knowair/24_24/all/his.npz \
  --precipitation-meters /path/to/KnowAir.npy \
  --city-file /path/to/city.txt \
  --project-root . \
  --out-root data/aeromosaic_bundles
```

For a software-workflow test, point the six `regions` fields in a copied YAML to the generated regional directories, validate them, and inspect the resulting dry-run commands. Do not compare such a run with the historical main-table numbers. This utility uses the final observation in the historical window for wind, a 600-km cutoff, cosine alignment at least 0.25, at most three incoming parents, cycle removal, and binary destination-by-source export. By contrast, the frozen BTH graph is weighted and the five other frozen regional graphs come from the legacy target-date-mean-wind canonical builder. `--max-groups 1` is available for disposable format checks only.

The date loop uses the previous calendar day's eight historical steps as input and the target day's eight-step mean PM2.5 as the label. Because the first timestamp has no preceding day, it emits 1,460 target dates (2015-01-02 through 2018-12-31): 730 train, 365 validation, and 365 test days. SDP has 15 cities (no Dezhou). The builder now emits every field required by both trainer and evaluator, including `parent_count_path` and `city_names_path`. Exact parity with the two historical experiment-time DAG variants is a separate provenance question, not a software-interface requirement.

Do not replace the linked historical assets with KnowAir-V2 or a different terrain product and describe it as the camera-ready input without author confirmation.

## 2. Check the release and data contract

The data-free check reads files only; `--check-cli` starts the model scripts in help mode and does not load data or weights:

```bash
python scripts/smoke_test_data_free.py --check-cli
```

Validate each configured frozen bundle with the generic contract. Do not pass `--paper-protocol`, because that option certifies the later unified binary/last-historical reconstruction rather than the archived main-table assets:

```bash
python scripts/validate_data_contract.py --bundle /path/to/BTH_bundle --project-root /path/to/aeromosaic --region BTH
python scripts/validate_data_contract.py --bundle /path/to/YRD_bundle --project-root /path/to/aeromosaic --region YRD
python scripts/validate_data_contract.py --bundle /path/to/SDP_bundle --project-root /path/to/aeromosaic --region SDP
python scripts/validate_data_contract.py --bundle /path/to/ZHD_bundle --project-root /path/to/aeromosaic --region ZHD
python scripts/validate_data_contract.py --bundle /path/to/CJD_bundle --project-root /path/to/aeromosaic --region CJD
python scripts/validate_data_contract.py --bundle /path/to/CYB_bundle --project-root /path/to/aeromosaic --region CYB
```

Expected split layout and array shapes are described in `docs/data_contract.md`.

## 3. Inspect then run the main experiment

First inspect all 18 train/evaluation commands without executing them:

```bash
python scripts/run_camera_ready_oracle.py --config configs/paths.yaml --dry-run
```

If the paths and compute allocation are correct, run sequentially:

```bash
python scripts/run_camera_ready_oracle.py --config configs/paths.yaml
```

The runner reproduces the archived April 16 `abl_AC` command line: regions BTH/YRD/SDP/ZHD/CJD/CYB; seeds 42/123/456; `pipeline/train_balanced_gc_ablation.py`; `pipeline/eval_dagresidual_ablation.py`; precipitation bundles; DAG residual; root-region fusion; root-aware loss; LoRA; three epochs; learning rate `1e-4`; gradient accumulation 2; and `--ablation_mask 0,11,12` during training and evaluation. It resumes the archived behavior by skipping an existing final adapter or completed test summary rather than overwriting it.

Each run writes training logs/configuration and test summaries beneath:

```text
<output_root>/abl_AC/<REGION>_seed<SEED>/
  final_adapter/
  run_config.json
  train.log
<output_root>/abl_AC/<REGION>_seed<SEED>_eval_test/
  eval.log
  test/summary.json
  test/predictions.csv
```

## 4. Aggregate the six-region results

After all 18 evaluations succeed:

```bash
python scripts/aggregate_camera_ready.py --runs-root /path/to/aeromosaic-results --out /path/to/aeromosaic-results/camera_ready_aggregate.json
```

The aggregator requires all six regions and all three seeds, reports population standard deviation (`ddof=0`), and verifies that every rounded Micro-F1 and Macro-F1 mean matches the published main table. It does not compare or combine baseline outputs.

## Reproduction boundary

Eighteen archived test summaries were checked during release preparation, and their per-seed statistics are preserved in `provenance/main_table_abl_AC_aggregate.json`; the individual historical summary files are not part of this source package and are not required for a new run. The runner mirrors their training/evaluation path. No end-to-end result was freshly regenerated during package validation, and baseline code/results are outside this release scope.

## 5. Matched BTH no-oracle diagnostics (single runs)

The October 4 paper appendix reports three matched parent-information protocols for BTH. `AeroMosaic` uses the numerical-precipitation variant in these controls. None uses target-day true parent labels. These are single runs, not three-seed estimates. Their `8,14` raw input shape differs from the primary oracle runner's `8,13`; do not reuse the oracle recipe unchanged.

The source for these builders is included at `data/bth_no_oracle_source_8x14`. It contains 730/365/365 train/validation/test daily groups and all indexed paths are relative to the release root. Validate it before deriving either control:

```bash
python scripts/validate_bth_no_oracle_bundle.py --release-root .
```

Two derived-data construction entry points are provided:

```bash
# No-parent: remove the upstream parent prompt field.
"$PY" diagnostics/build_e10_noparent_bundle.py \
  --source-data-dir data/bth_no_oracle_source_8x14 \
  --out-dir /path/to/derived/bth_no_parent

# Historical upstream: include each parent's last observed PM2.5 value
# from the input window (not target-day observations).
python diagnostics/build_e7_lag1_pm25_bundle.py \
  --source_data_dir data/bth_no_oracle_source_8x14 \
  --out_dir /path/to/derived/bth_historical_upstream
```

The authoritative release entrypoints are `pipeline/train_balanced_gc_ablation.py` for fitting and `pipeline/eval_dagresidual_ablation.py` for the no-parent and historical-upstream evaluations. The predicted-parent condition instead uses `diagnostics/eval_predicted_rollout.py`, as shown in the executable recipe below. Older private names such as `train_balanced_gc.py` and `eval_yrd_graph_slices_v2_dagresidual.py` are not release commands. Use the paper's single-run seed `2024`, `--raw_ts_shape 8,14`, three epochs, root-aware loss (`root_weight=1.5`, class additions `3:2.0,4:3.0`, max root sample weight `4`), root-region environment fusion (weight `0.8`, exclude self), DAG residual width 256 / alpha initialization `-3` / indegree normalization, LoRA `r=16, alpha=32, dropout=0.05`, learning rate `1e-4`, warmup `0.03`, batch size 1, gradient accumulation 2, max input length 512, mean pooling and bf16. Select by `joint_score`, with `best_macro_tol=0.003` and `joint_stage_bonus=10`. These values were checked against the archived run configurations and the October 4 appendix; the release includes the prepared BTH source bundle but not archived adapters or model weights.

For an executable no-oracle recipe, run from the release root and set the ChatTS checkpoint and a fresh output directory. The included prepared BTH numerical-precipitation source bundle has `8x14` tensors and train/validation/test group indexes. The commands below reproduce the archived single-run configuration; use a CUDA GPU with enough memory for ChatTS.

```bash
PY="$(command -v python)"
MODEL="/path/to/ChatTS-8B-1103"
ORACLE_SOURCE="$PWD/data/bth_no_oracle_source_8x14"
OUT="/path/to/new/bth_no_oracle_runs"
NOPARENT_DATA="$OUT/bth_noparent_bundle"
NOPARENT_RUN="$OUT/E10_noparent_seed2024"
WARM_ROLLOUT="$OUT/warmup_rollout_trainval"
PRED_DATA="$OUT/predicted_only_bundle"
PRED_RUN="$OUT/E10_predicted_only_seed2024"

"$PY" diagnostics/build_e10_noparent_bundle.py \
  --source-data-dir "$ORACLE_SOURCE" --out-dir "$NOPARENT_DATA"

COMMON=(--model_name_or_path "$MODEL" --train_split train --eval_split val \
  --bf16 --root_aware_loss --root_label_extra 3:2.0,4:3.0 \
  --max_root_sample_weight 4.0 --root_region_env_fusion \
  --root_region_exclude_self --num_train_epochs 3 --learning_rate 1e-4 \
  --gradient_accumulation_steps 2 --logging_steps 10 --auto_num_labels \
  --root_weight 1.5 --root_region_env_weight 0.8 --seed 2024 \
  --dag_residual_enable --dag_residual_alpha_init -3.0 \
  --dag_residual_hidden_dim 256 --raw_ts_shape 8,14)

"$PY" pipeline/train_balanced_gc_ablation.py \
  "${COMMON[@]}" --data_dir "$NOPARENT_DATA" --out_dir "$NOPARENT_RUN"

"$PY" diagnostics/eval_predicted_rollout.py \
  --model_name_or_path "$MODEL" \
  --adapter_dir "$NOPARENT_RUN/final_adapter" \
  --oracle_data_dir "$ORACLE_SOURCE" --noparent_data_dir "$NOPARENT_DATA" \
  --out_dir "$WARM_ROLLOUT" --eval_splits train,val \
  --raw_ts_shape 8,14 --bf16 --trust_remote_code

"$PY" diagnostics/prepare_e7_scheduled_bundle.py \
  --oracle_data_dir "$ORACLE_SOURCE" --rollout_dir "$WARM_ROLLOUT" \
  --out_dir "$PRED_DATA" --seed 2024 \
  --train_pred_prob 1.0 --val_pred_prob 1.0

"$PY" pipeline/train_balanced_gc_ablation.py \
  "${COMMON[@]}" --data_dir "$PRED_DATA" --out_dir "$PRED_RUN"

"$PY" diagnostics/eval_predicted_rollout.py \
  --model_name_or_path "$MODEL" \
  --adapter_dir "$PRED_RUN/final_adapter" \
  --oracle_data_dir "$ORACLE_SOURCE" --noparent_data_dir "$NOPARENT_DATA" \
  --out_dir "$PRED_RUN/predicted_rollout_test" --eval_splits test \
  --raw_ts_shape 8,14 --bf16 --trust_remote_code
```

The first fit is a no-parent warm-up model; its train/validation predictions populate parent prompts for the second fit. The final test evaluator predicts cities level-by-level in the daily DAG and replaces the upstream prompt block, so target-date true parent labels are not passed to the model. The paper explicitly cautions that this one-day group rollout is not a complete temporal recursive-forecasting analysis. For the no-parent and historical-upstream rows, use the builders above and the same `COMMON` fit flags, then evaluate with `pipeline/eval_dagresidual_ablation.py` using `--eval_splits test --raw_ts_shape 8,14 --bf16 --trust_remote_code`.

The archived BTH AeroMosaic test summaries were checked against the paper's table in Appendix `Matched No-Oracle Parent-Information Protocols` (Micro-F1/Macro-F1): no parent `0.5334/0.3049`, predicted only `0.5128/0.2458`, historical upstream PM2.5 `0.5825/0.3462`. This is a consistency check of existing outputs, not a fresh reproduction. The paper's MGSFormer and TimeMixer values are `0.6303/0.5091` and `0.5545/0.4083` (no parent), `0.6272/0.5070` and `0.5410/0.4166` (predicted only), and `0.6318/0.5342` and `0.5557/0.4052` (historical upstream). Their source implementations are not redistributed here; see the pinned provenance and remaining per-method reproduction gap in `docs/baseline_reproduction.md`.

## 6. Baseline reproduction boundary

The 13 main-table methods, CauAir source commit, and SHA-256 fingerprints for their entrypoints are recorded in `docs/baseline_reproduction.md`. The source implementations remain external. The hashes pin entrypoints but not each method's full dependency tree or the experiment invocation. A per-method command/config/data/seed/selection mapping and execution evidence are still needed before claiming the paper's baseline rows are reproducible from this release. Baseline numerical results are reported paper results, not results regenerated by this package.


## 7. Audit update — exact historical BTH DAG alignment

The detailed [data and DAG provenance audit](data_and_dag_provenance_audit.md) records deterministic checks against the preserved BTH daily reference bundle. On all 1,460 dates, the **legacy audit utility** `scripts/build_bth_group_bundle.py` reproduced adjacency (including weights/orientation), levels, roots, city ordering, and complete prompt JSON. This establishes exact historical BTH DAG/prompt alignment against that reference. It does not mean that the prospective unified builder `scripts/prepare_camera_ready_data.py` recreates the main-table BTH bundle, nor does it establish the undocumented provenance of the reference or parity for other regions. The available `his.npz` reproduces target labels, historical PM2.5, and hour, but not the full archived meteorological input tensor; raw-data parity remains open.


## Historical-bundle protocol warning (2026-10-16)

Do not use the preserved YRD/SDP/ZHD/CJD/CYB `canonical_v2` bundles as an exact-equality oracle for `scripts/prepare_camera_ready_data.py`: their DAG builder used target-day mean wind and hybrid target-day weather inputs, while this release script follows the manuscript's last-historical wind cue. A full 1,460-date comparison found zero matching edge patterns in each of those five regions. The separate BTH reference did match the release DAG and prompts, but the available `his.npz` does not reconstruct every historical input channel. See `docs/data_and_dag_provenance_audit.md` for the audit and unresolved binary-versus-weighted adjacency notation.
