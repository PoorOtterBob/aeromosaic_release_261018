# AeroMosaic camera-ready code release (staged)

This is a curated, de-anonymized source package for **AeroMosaic: Transport-Aware Multimodal Evidence Fusion for Atmospheric Pollution Risk Inference**, kept separate from the mutable experiment tree. It is staged for public release review; the project-code license and baselines are intentionally unresolved/excluded.

## Main-table provenance is authoritative (2026-10-17)

The public main entry point is tied to the archived experiment that generated the paper's six-region means: `abl_AC` from 2026-04-16. It invokes `pipeline/train_balanced_gc_ablation.py` and `pipeline/eval_dagresidual_ablation.py`, uses seeds `42/123/456` configured in `configs/paths.yaml`, enables precipitation prompts, DAG residual propagation, root-region fusion, root-aware loss and LoRA, and applies `--ablation_mask 0,11,12` during both training and evaluation. Eighteen archived evaluation summaries were audited outside this source package; their per-seed statistics and all 24 four-decimal mean/SD checks are recorded in `provenance/main_table_abl_AC_aggregate.json`. The individual historical summaries are not required to run the software and are not redistributed.

A **bundle** is simply the processed software input produced by a data builder: one regional directory containing train/validation/test JSONL indices plus per-date tensors, DAG metadata, city names, parent counts and prompts. The six historical experiment-time bundles named in `configs/paths.example.yaml` are the provenance authority for exact main-table reruns. They are not interchangeable with bundles produced by the later `scripts/prepare_camera_ready_data.py` reconstruction. The BTH frozen graph uses an earlier weighted, last-historical-wind implementation; the YRD/SDP/ZHD/CJD/CYB frozen graphs use the legacy canonical binary implementation with target-date mean wind, an approximately 300-km candidate rule, a 60-degree alignment rule and no BTH-style three-parent cap. The paper and this release disclose that distinction rather than relabeling the historical assets as outputs of one unified builder.

The later unified binary/last-historical builder remains available as a reconstruction and prospective-data utility. It does **not** reproduce the five-region main-table graphs and is not called by the main-table runner. The root-aware training objective implemented by the archived code is `sum_i(w_i CE_i) / sum_i(w_i)`. `scripts/aggregate_camera_ready.py` reads the archived output layout, uses population standard deviation (`ddof=0`), and verifies all rounded main-table means. No full experiment was rerun during this alignment revision.

See `provenance/README.md` for the preserved launcher hash, byte-identical train/evaluation source hashes, the per-seed aggregate, and the successful checks for all 12 main-table means and 12 population standard deviations.

## Scope and reproducibility boundary

The package contains AeroMosaic training/evaluation code, DAG/data-preparation and diagnostic utilities, a camera-ready six-region/three-seed runner and aggregator, a curated ChatTS source tree, and the self-contained BTH `8x14` prepared bundle needed by the matched no-oracle diagnostics. It does not redistribute the six frozen main-table bundles, general KnowAir/PM2.5-GNN processed data or station metadata, the legacy terrain raster, ChatTS/Qwen model weights, trained LoRA adapters, experiment outputs, caches, or copied baseline implementations.

To prepare software inputs, obtain the processed KnowAir [`his.npz` from the CauAir repository](https://github.com/PoorOtterBob/CauAir/tree/main/data_knowair/24_24/all), the original `KnowAir.npy`, and the exact ChatTS checkpoint separately, subject to upstream terms. The upstream [PM2.5-GNN Dataset instructions](https://github.com/shuowang-ai/PM2.5-GNN#dataset) point to the original KnowAir archive ([Zenodo DOI 10.5281/zenodo.23339627](https://doi.org/10.5281/zenodo.23339627)) and list Google Drive and Baidu Cloud mirrors (Baidu access code `t82d`). `scripts/prepare_camera_ready_data.py` consumes CauAir's processed `his.npz` together with aligned raw precipitation from `KnowAir.npy` and builds all six runnable regional bundles. No additional raw-to-`his.npz` conversion is required for this packaged workflow. Do not substitute KnowAir-V2 and describe it as the camera-ready input.

The upstream PM2.5-GNN repository also contains the supporting geospatial files: [`city.txt`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/data/city.txt) (city coordinates/order), [`station_info.csv`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/data/station_info.csv) (monitoring-station metadata), and [`altitude.npy`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/data/altitude.npy) (elevation grid). Its [`dataset.py`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/dataset.py) shows its upstream raw-array loading, meteorological feature selection, wind-speed/direction and time-feature construction, and normalization; [`graph.py`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/graph.py) shows its coordinate/altitude-based graph setup. The included six-region preparation script reads `city.txt` for city names, indices, and coordinates; it does not read `station_info.csv` or `altitude.npy`, and it does not apply a terrain filter. The upstream files are pointers, not proof of byte identity with experiment assets. The elevation product lineage and redistribution terms remain unresolved. The script requires a pre-existing normalized `his.npz`; see `docs/reproduction.md` for its exact boundary and DAG construction. Validate generated bundles against `docs/data_contract.md`.

The exact ChatTS checkpoint is `bytedance-research/ChatTS-8B` at revision `7216a84034063c2d394901fe45d20a1e8d773a1f`; see the model page and upstream source below. Checkpoint and processed data are not included.

The supplied runner mirrors the archived `abl_AC` configuration rather than the later unified reconstruction. It covers BTH/YRD/SDP/ZHD/CJD/CYB, the 2015--2016/2017/2018 split, seeds 42/123/456, and mask `{0,11,12}`. Archived BTH no-oracle controls remain separate single-run diagnostics with raw input shape `8x14`; they must not be mixed with the primary `8x13` oracle protocol. Their prepared source is distributed at `data/bth_no_oracle_source_8x14`; its indexed paths are release-root-relative and all referenced files are inside that directory. A fresh environment install and a fresh 18-run regeneration remain unverified. Baseline source commit and entrypoint hashes are recorded in `docs/baseline_reproduction.md`, but their paper-specific command/data mapping is not verified and implementations are not redistributed.

## Upstream assets

- ChatTS source: <https://github.com/NetManAIOps/ChatTS>
- ChatTS-8B model: <https://huggingface.co/bytedance-research/ChatTS-8B>
- PM2.5-GNN / KnowAir project: <https://github.com/shuowang-ai/PM2.5-GNN>
- CauAir source and processed KnowAir `his.npz`: <https://github.com/PoorOtterBob/CauAir/tree/main/data_knowair/24_24/all>

The `ChatTS/` directory contains a curated source subset, its upstream README, requirements, and LICENSE; it does not contain model weights or Git metadata. Every included file that also exists at upstream commit `a16ca1a7bd2d0cbe1dd40af37cb5658e7008357e` is byte-identical after the final source audit. Unrelated upstream directories are intentionally omitted. AeroMosaic does not patch ChatTS's data-generation or vLLM utilities; the paper-aligned pipeline uses the external checkpoint through Transformers and implements AeroMosaic-specific logic under `pipeline/`. See `ChatTS/AEROMOSAIC_INTEGRATION.md`. Review all third-party license obligations before public distribution. The paper's model repository/revision is `bytedance-research/ChatTS-8B` / `7216a84034063c2d394901fe45d20a1e8d773a1f`; model-card terms are separate. Historical processed KnowAir/station/terrain licensing is unresolved; those general assets remain excluded. The separately bundled BTH no-oracle prepared input is included at the author's direction and is documented as preserved experiment-time input rather than a public-raw-data reconstruction.

## Environment and execution

The research scripts were developed on Linux with CUDA and a large-memory NVIDIA GPU. `environment.yml` pins direct dependencies and `configs/paths.example.yaml` provides a portable template, including the authoritative seeds `[42, 123, 456]`; local `configs/paths.yaml` is ignored. The runner reads and validates those YAML seeds. Use Python 3.11 and CUDA 12.4, then follow `docs/reproduction.md`.

The main-table training/evaluation entry points are `pipeline/train_balanced_gc_ablation.py` and `pipeline/eval_dagresidual_ablation.py`; orchestration and aggregation are `scripts/run_camera_ready_oracle.py` and `scripts/aggregate_camera_ready.py`. Other pipeline files are retained for diagnostics and historical comparison but are not the main-table authority. A fresh clean-environment CLI smoke and a fresh 18-run regeneration remain unverified; missing dependencies may surface before model/data loading. Processed observations and model weights are intentionally external.

## Citation

```bibtex
@inproceedings{zheng2026aeromosaic,
  title     = {AeroMosaic: Transport-Aware Multimodal Evidence Fusion for Atmospheric Pollution Risk Inference},
  author    = {Zheng, Muyang and Ma, Jiaming and Zhang, Zongyu and Wen, Qingsong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Update this citation with final proceedings metadata and the public repository URL once available.

## Current verification and release status

This package passed manifest integrity, Python syntax, data-free smoke, paper-protocol unit tests, full-shape synthetic six-region construction, BTH/YRD generic and paper-protocol validation, three-seed runner dry-run, and aggregate-statistics checks. The validator also rejects non-integer/out-of-range labels, non-finite tensors, malformed graph arrays and region-size mismatches. These are software/interface tests; no real training or paper-result regeneration was performed.

A clean environment and model-dependent CLI smoke remain unverified. Depending on the starting environment, missing packages such as PyTorch, PEFT or other pinned dependencies may fail before model/data loading. The historical one-group execution check used external dependencies available at that time and is not evidence of a clean install today. No research environment was modified during final packaging.

Historical BTH DAG/prompt parity applies specifically to `scripts/build_bth_group_bundle.py`, the legacy audit utility. The prospective six-region builder uses a different binary/last-historical contract, and the five non-BTH main-table bundles use their disclosed historical target-day-weather protocol. See [the current provenance statement](docs/data_and_dag_provenance_audit.md).

The package is publicly released at https://github.com/miya1102/aeromosaic_release_261018. Project-code licensing and complete paper-specific baseline reproduction remain intentionally deferred. Exact historical main-table AeroMosaic reruns still require the six external frozen bundles and the external ChatTS checkpoint; the matched BTH no-oracle source bundle is included. See `RELEASE_CHECKLIST.md`, `docs/reproduction.md`, `docs/baseline_reproduction.md`, and `MANIFEST.sha256`.
