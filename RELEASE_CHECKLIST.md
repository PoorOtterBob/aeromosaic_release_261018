# AeroMosaic camera-ready release checklist

Current authoritative status: 2026-10-18. The public release is https://github.com/miya1102/aeromosaic_release_261018. This checklist replaces earlier staged-candidate notes. It records what the package establishes and what remains external; it is not evidence that experiments were freshly rerun.

## Paper and implementation alignment

- [x] Primary protocol covers BTH, YRD, SDP, ZHD, CJD and CYB.
- [x] Train/validation/test periods are 2015--2016 / 2017 / 2018.
- [x] The camera-ready runner uses seeds 42, 123 and 456 and mask `{0,11,12}`.
- [x] Runner defaults match the archived `abl_AC` launcher and paper hyperparameters.
- [x] Training and evaluation implement the documented root fusion, root-aware loss and DAG residual path.
- [x] The oracle parent-label prompt is explicitly identified as privileged target-date information.
- [x] The historical DAG split is disclosed: weighted last-historical BTH and binary target-day-mean-wind DAGs for the other five regions.
- [x] The prospective unified builder is clearly separated from the historical frozen main-table bundles.
- [x] AeroMosaic main-table aggregates in `provenance/main_table_abl_AC_aggregate.json` match the camera-ready paper.
- [ ] A fresh six-region, three-seed experiment regeneration was not performed.

## Source and package integrity

- [x] The release contains training, evaluation, data/DAG builders, diagnostics, runner, aggregator and the required ChatTS source subset.
- [x] The ChatTS checkpoint revision and upstream source comparison are documented; every vendored file shared with pinned commit `a16ca1a7...` is byte-identical.
- [x] `MANIFEST.sha256` covers every distributed file except itself.
- [x] Archive traversal and link checks pass.
- [x] All distributed Python files parse successfully.
- [x] Data-free smoke and paper-protocol unit tests pass.
- [x] Generated caches and `__pycache__` directories are excluded.
- [x] The archive root and outer filename use the same 20261017 release name.

## Public data and construction path

- [x] README links the public CauAir `his.npz`, PM2.5-GNN `city.txt`, original KnowAir source and ChatTS checkpoint.
- [x] `scripts/prepare_camera_ready_data.py` builds all six prospective regional bundles.
- [x] The builder emits every trainer/evaluator-required field, including parent counts and city names.
- [x] BTH and YRD have been exercised with a full-shape synthetic input through build, generic validation and paper-protocol validation.
- [x] The validator checks region size, shapes, finiteness, label dtype/range, graph integrity, roots/levels, parent counts, city names and prompts.
- [x] The runner dry-run emits 18 training and 18 evaluation commands and reads the three YAML seeds.
- [x] The aggregator requires all six regions and three seeds and uses population standard deviation (`ddof=0`).
- [ ] The authoritative raw `KnowAir.npy` to processed `his.npz` writer was not recovered; the linked processed file is the supported public input.

## Historical-result boundary

- [x] Exact main-table reruns are documented as requiring the six external experiment-time frozen bundles.
- [x] Historical BTH DAG/prompt parity is attributed specifically to the legacy audit utility, not to the prospective unified builder.
- [x] The five-region historical target-day-weather/DAG protocol is documented separately.
- [x] The 18 historical summaries are described as packaging-time evidence; their aggregate statistics are preserved, but the individual files are not claimed to be distributed.
- [x] The no-oracle commands consistently use the release ablation trainer/evaluator and predicted-rollout entrypoint.
- [x] The separate `8x14` BTH no-oracle source bundle is distributed, self-contained, and validated across all 1,460 daily groups.
- [x] The de-anonymized public GitHub repository URL is documented in the paper and README.

## Environment and execution status

- [x] No true target-city future PM2.5 enters the model input.
- [x] No real training was performed during final package validation.
- [x] Existing research environments were not modified.
- [ ] A clean install and model-dependent CLI smoke remain unverified.
- [ ] A CUDA machine with sufficient memory and the external ChatTS checkpoint is required for real training.

## Explicitly deferred by the author

- [ ] Project-code license selection and final third-party legal review.
- [ ] Complete paper-specific baseline commands, configurations, checkpoints and execution evidence.

## Release assessment

The source package is suitable as a transparent camera-ready software release: its prospective data/software path is closed and tested without training, and its paper-aligned historical execution path is accurately documented. It must not be advertised as a from-public-raw-data reproduction of every main-table row. Exact historical AeroMosaic reruns require the original frozen bundles; baseline and license work remain separate deferred items.
