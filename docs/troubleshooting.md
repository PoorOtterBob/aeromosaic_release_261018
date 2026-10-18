# Troubleshooting

This guide covers only files and commands included in this release. Start with `README.md`, `docs/reproduction.md`, and `docs/data_contract.md`. The dated `docs/*baseline_precip_ac_20260417.md` notes are retained as historical experiment records; they are not runnable scripts or current release recipes.

## Environment and model

### `torch.cuda.is_available()` is false

This release expects Linux, Python 3.11, CUDA 12.4-compatible PyTorch, and a supported NVIDIA driver. Check `nvidia-smi` and the PyTorch/CUDA compatibility for the environment you installed. The included `environment.yml` has not been verified by a clean installation, so do not interpret this checklist as a tested environment setup.

### CLI smoke check reports a missing package such as `peft`

The smoke command checks import/help behavior; it does not install dependencies or test a model run. Install the declared dependencies in a new isolated environment and retry:

```bash
conda env create -f environment.yml
conda activate aeromosaic-repro
python scripts/smoke_test_data_free.py --check-cli
```

As of the latest package audit, a clean install was not completed and the existing server environment lacked `peft`. This release therefore does not claim that the smoke check passes in a fresh environment.

### Model loading fails

Set `model_dir` in your local `configs/paths.yaml` to the downloaded ChatTS-8B-1103 checkpoint at the revision documented in `README.md` and `docs/reproduction.md`. The weights are not included. Review the model-card terms before downloading. There is no `scripts/verify_model_artifacts.py` in this release; inspect the checkpoint files and use the trainer/evaluator CLI help for supported options.

## Data

### A regional bundle is missing or validation fails

Use processed `data_knowair/24_24/all/his.npz` from the linked CauAir repository, original `KnowAir.npy` from the PM2.5-GNN dataset instructions, and `city.txt`. This release includes `scripts/prepare_camera_ready_data.py` to turn those inputs into six regional bundles. Follow `docs/reproduction.md` for the exact paths and DAG limitations; do not substitute KnowAir-V2 or undocumented station/terrain assets. Then validate each generated bundle:

```bash
python scripts/validate_data_contract.py \
  --bundle /path/to/region_bundle \
  --project-root /path/to/aeromosaic \
  --region BTH
```

Replace `BTH` with `YRD`, `SDP`, `ZHD`, `CJD`, or `CYB` as needed. The validator is read-only and checks indexed splits, referenced files, prompt count, array shapes, and labels. Bundle creation is performed by `scripts/prepare_camera_ready_data.py`; neither utility establishes that an input dataset is legally reusable.

### Tensor shape mismatch

The primary oracle protocol expects `raw_ts` shape `(N, 8, 13)`: historical PM2.5, ten meteorological channels, and two time channels. The archived BTH no-oracle controls use `(N, 8, 14)` and are separate protocols; see `docs/reproduction.md`. Do not convert between them by guessing a channel mapping.

## Running the primary protocol

### Inspect commands without launching training

```bash
python scripts/run_camera_ready_oracle.py \
  --config configs/paths.yaml --dry-run
```

The dry run prints train/evaluation commands but does not verify dependencies, data accessibility, or model loading. The actual 18-run recipe and output structure are in `docs/reproduction.md`.

### CUDA out-of-memory or training is slow

Check available memory with `nvidia-smi`. The documented recipe uses group batch size 1 and gradient accumulation 2; reduce batch size only if you also document that this changes the protocol. No automatic GPU-selection orchestrator or `scripts/run_full_pipeline.sh` is included in this release.

### Expected outputs are missing

Training artifacts are written under `<output_root>/abl_AC/<REGION>_seed<SEED>/`; evaluation artifacts are written under `<output_root>/abl_AC/<REGION>_seed<SEED>_eval_test/`. The runner resumes conservatively: it skips training when `final_adapter/` exists and skips evaluation when `test/summary.json` exists. If a directory exists without the corresponding completion artifact, the runner continues that stage and writes its log there. Check `train.log` or `eval.log` for the first error.

After all 18 runs finish, aggregate them with:

```bash
python scripts/aggregate_camera_ready.py \
  --runs-root /path/to/aeromosaic-results \
  --out /path/to/aeromosaic-results/camera_ready_aggregate.json
```

## No-oracle controls and baselines

The BTH no-oracle recipes and their input assumptions are documented in `docs/reproduction.md`; they require an externally prepared BTH `(N, 8, 14)` source bundle and the external model checkpoint. The dated A+C+T notes describe an older private experiment tree and its old paths are intentionally not presented as current commands.

Baseline provenance and the limits of reproducing baseline rows are in `docs/baseline_reproduction.md`. Baseline implementations are not included, and the per-method camera-ready command/config/data mapping is not verified. This package cannot run those baselines by following its own scripts.

## Validation scope

`scripts/smoke_test_data_free.py` and `scripts/validate_data_contract.py` are the current data-free and bundle-validation entry points. Earlier checklist text may describe older commands or intermediate checks; use the latest dated verification updates in `RELEASE_CHECKLIST.md` for current status. Neither a dry run nor a data-contract pass demonstrates fresh installation or reproduces paper metrics.
