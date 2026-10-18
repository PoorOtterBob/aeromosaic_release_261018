# Main-table paper-to-code alignment (2026-10-17)

Authority order for this release:

1. the published six-region main-table means;
2. the archived 2026-04-16 `abl_AC` launcher, run configurations and 18 test summaries;
3. the release runner and documentation;
4. manuscript method wording.

No training was rerun for this audit. The conclusion is based on preserved
commands, source hashes, run configurations, data paths and result summaries.

## Exact main-table chain

```text
run_ablation_v2.sh
  -> abl_AC = precip | DAG=true | mask=0,11,12 | LoRA=true
  -> train_balanced_gc_ablation.py
  -> eval_dagresidual_ablation.py
  -> six frozen *_dag_group_bundle_precip* bundles
  -> 6 regions x seeds {42,123,456}
  -> 18 test/summary.json files
```

All 12 rounded Micro-F1/Macro-F1 means and all 12 population standard deviations
from these summaries equal the values in the corrected paper main table. The
archived train/evaluation files and their release
copies are byte-identical by SHA-256. `scripts/run_camera_ready_oracle.py` now
calls those files with the archived `abl_AC` flags, and
`scripts/aggregate_camera_ready.py` verifies the rounded means and error bars.

| Paper/result fact | Authoritative implementation or artifact | Status |
|---|---|---|
| Six regions and three seeds | BTH/YRD/SDP/ZHD/CJD/CYB; 42/123/456 | Matched |
| Main numerical branch | `--ablation_mask 0,11,12` in train and evaluation | Matched |
| Precipitation prompt path | Frozen precipitation bundles | Matched at artifact level; raw regeneration is not complete |
| Oracle parent-label prompts | Stored prompts read by the archived dataset path | Matched; privileged upper-bound input |
| DAG residual | `--dag_residual_enable`, alpha `-3`, hidden width `256` | Matched |
| Root-region fusion | enabled, weight `0.8`, self excluded | Matched |
| Root-aware loss | base `1.5`, class additions `3:2.0,4:3.0`, cap `4.0` | Matched |
| Loss normalization | `sum_i(w_i CE_i) / sum_i(w_i)` | Paper equation corrected to code |
| LoRA and training length | LoRA enabled; three epochs; learning rate `1e-4` | Matched |
| Main-table statistics | 18 archived test summaries | All 12 means and 12 population SDs match at four decimals |

## Frozen DAG provenance

The six main-table bundles share the same training interface but not one
byte-identical graph-construction protocol.

| Region group | Frozen experiment-time DAG provenance |
|---|---|
| BTH | Weighted graph from last-historical wind; parent selection/capping and cycle removal |
| YRD/SDP/ZHD/CJD/CYB | Legacy canonical binary graph from target-date mean wind; approximately 300-km candidates, 60-degree alignment, no BTH-style three-parent cap, topological city reordering |

The later `scripts/prepare_camera_ready_data.py` builder uses a unified binary,
last-historical-wind protocol. It is retained for reconstruction and prospective
data preparation, but it is not the source of the published main-table graphs.
The main runner therefore validates the generic bundle contract and deliberately
does not require the later `--paper-protocol` manifest.

## Remaining boundaries

1. Model weights are external. Exact historical main-table reruns additionally
   require the experiment-time bundles; ordinary packaged-workflow bundles can
   be rebuilt from the linked CauAir processed KnowAir asset.
2. CauAir's processed `his.npz` closes the runnable software input path. The
   older raw KnowAir-to-historical-13-channel conversion remains relevant only
   to byte-level provenance of the experiment-time inputs.
3. The archived summaries establish result provenance, but a clean-environment
   end-to-end rerun of all 18 experiments has not been completed.
4. Population standard deviation is recomputed from the archived seeds. The
   2026-10-17 audit corrected three stale manuscript error bars: SDP Micro-F1
   `0.0024 -> 0.0025`, ZHD Micro-F1 `0.0065 -> 0.0064`, and CJD Macro-F1
   `0.0336 -> 0.0331`. No mean changed.
5. Baseline command/data mapping and project-license decisions are tracked
   separately and do not alter the AeroMosaic main-table chain above.

The release can therefore claim exact archived provenance for the paper's
main-table statistics and an executable matching command path, but not complete
raw-data regeneration or a fresh independent rerun.
