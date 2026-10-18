# Data and DAG provenance: current authoritative status

Audit status: 2026-10-17. This document reports the final release interpretation. It intentionally omits superseded intermediate decisions so that it cannot be used as an alternative execution guide. No private arrays are distributed in this source package.

## Public packaged-workflow inputs

The public [CauAir processed KnowAir directory](https://github.com/PoorOtterBob/CauAir/tree/main/data_knowair/24_24/all) contains `his.npz`. The copy examined during packaging had SHA-256 `ac360c2b023757ee8c691d7d3e222cc800eac33f232a615ebb3d2dd0e01ecb8f`, keys `data`, `mean`, and `std`, and `data` shape `(11688,184,13)`. The prospective release builder consumes this processed asset directly, together with aligned unstandardized precipitation in metres and the upstream [PM2.5-GNN `city.txt`](https://github.com/shuowang-ai/PM2.5-GNN/blob/main/data/city.txt).

The 13 channels are documented as PM2.5; temperature at 2 m; boundary-layer height; K index; RH950; surface pressure; total precipitation; u950; v950; wind speed; wind direction; hour; weekday. The authoritative raw `KnowAir.npy` to `his.npz` writer, including all featurewise moments, fit period, missing-value treatment and derived-wind choices, was not recovered. Therefore the release supports the linked processed input but does not claim byte-level reconstruction of `his.npz` from the raw archive.

## Three distinct assets and claims

Do not conflate the following:

1. **Historical main-table frozen bundles.** These are the experiment-time inputs used by the archived `abl_AC` runs. They are external and are referenced by `configs/paths.example.yaml`. The release runner validates their generic trainer/evaluator contract; it does not require the prospective builder's manifest.
2. **Legacy BTH audit utility.** `scripts/build_bth_group_bundle.py` is retained to audit the historical BTH construction. It is not the six-region public builder.
3. **Prospective public builder.** `scripts/prepare_camera_ready_data.py` creates new six-region bundles from the linked processed inputs. It emits a binary destination-by-source DAG based on the final historical wind observation, a 600-km candidate radius, cosine alignment of at least 0.25, no more than three incoming parents, cycle removal, and all fields required by the trainer/evaluator. Validate these outputs with `--paper-protocol`. They are not substitutes for the historical main-table frozen bundles.

## Historical BTH verification

Against the preserved private BTH reference, the legacy BTH audit utility reproduced all 1,460 dates for city order, weighted adjacency and orientation, DAG levels, root mask, and complete prompt JSON. The reference had 730/365/365 train/validation/test records and input/label shapes `(13,8,13)` and `(13,)`.

The available public `his.npz` reproduced the historical target labels, PM2.5 channel and hour channel on all 1,460 dates. It did not reproduce every historical meteorological channel; total precipitation matched only 139 of 1,460 complete daily channel comparisons, and the remaining meteorological/weekday channels did not match as complete channels. Thus the legacy utility establishes BTH DAG/prompt alignment, not full raw-input parity.

## Historical YRD/SDP/ZHD/CJD/CYB verification

The recovered five-region frozen bundles used a different experiment-time protocol: target-day mean wind from the recorded forecast tensor, approximately 300-km geographic candidates, 60-degree directional alignment, no BTH-style three-parent cap, topological city reordering, and binary graph export. Their hybrid tensors combine historical PM2.5 with target-day weather channels.

An exhaustive comparison with the legacy last-historical BTH construction found zero identical edge patterns over 1,460 dates in each of YRD, SDP, ZHD, CJD and CYB. This is expected because the protocols differ; these assets must not be described as outputs of the prospective public builder.

For the first historical date, all five regions matched public `his.npz` on PM2.5, hour and target label, but none matched the complete 13-channel tensor. This was a one-date diagnostic, not full raw-input certification.

## Main-table resolution

The camera-ready paper and release disclose the historical split: weighted last-historical BTH versus binary target-day-mean-wind construction for the other five regions. The portable runner mirrors the archived `abl_AC` launcher, mask, loss and hyperparameters and consumes the six external frozen bundles. The 18 historical test summaries were inspected during packaging, and their per-seed statistics are preserved in `provenance/main_table_abl_AC_aggregate.json`; the individual summaries are not distributed.

Accordingly:

- the paper, archived implementation path and reported AeroMosaic aggregate values are aligned;
- the source package provides an executable prospective data-contract path;
- exact main-table reruns require the original external frozen bundles and model checkpoint;
- no full experiment was rerun during packaging;
- neither the prospective builder nor synthetic smoke tests are evidence that the historical main table was freshly reproduced.

## What the validator proves

`scripts/validate_data_contract.py` verifies the indexed files, required trainer/evaluator fields, region node count, shapes, numeric finiteness, label dtype/range, non-negative loop-free adjacency, roots/levels, parent counts, city names and prompt count. `--paper-protocol` additionally certifies the prospective binary/last-historical manifest, binary adjacency, level ordering and oracle prompt fields. Validation proves interface consistency, not source provenance or equality to the historical frozen bundles.
