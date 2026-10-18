# Baseline source provenance and reproduction boundary

The camera-ready paper compares AeroMosaic with 13 existing baselines. Their implementations are not copied into this release; this document records the upstream source provenance and the remaining experiment-protocol gap without redistributing any third-party code.

## Upstream snapshot

- Repository: <https://github.com/PoorOtterBob/CauAir>
- Verified source checkout commit: `73dae00ca6ad14abb15174a0a0286d500e868b94` (commit date 2026-05-12).
- The checkout was clean for source files; one bundled data file was modified locally and is irrelevant to the baseline entrypoint hashes below.
- To obtain that source snapshot independently:

```bash
git clone https://github.com/PoorOtterBob/CauAir.git
cd CauAir
git checkout 73dae00ca6ad14abb15174a0a0286d500e868b94
```

The source subtree maps the paper's methods as follows: DLinear → `experiments/dlinear`, CATS → `experiments/cats`, CrossGNN → `experiments/crossgnn`, DSFormer → `experiments/dsformer`, SOFTS → `experiments/softs`, SparseTSF → `experiments/sparsetsf`, TimeMixer → `experiments/timemixer`, TimeXer → `experiments/timexer`, UMixer → `experiments/umixer`, AirFormer → `experiments/airformer`, AirPhyNet → `experiments/airphynet`, GAGNN → `experiments/gagnn`, and MGSFormer → `experiments/mgsfformer`.

The SHA-256 values below were computed for each upstream `experiments/<method>/main.py` in the server checkout and matched the corresponding entrypoint copied in the server's dated baseline inventory. This pins only those entrypoint files to the cited checkout; it does not pin the rest of the dependency tree, configs, data transformation, or training invocation.

| Method | SHA-256 of `experiments/<method>/main.py` |
|---|---|
| DLinear | `f3fdd57e7944fe212e95cc0793f7be98e3bcb362e113c9fd80622c2792adda77` |
| CATS | `fd2f6a0cd2cc029aae9d19e4840bf39f8c30c3bc97afe02caa5a26ebdc5c824a` |
| CrossGNN | `113daf0de817b2d7f52fb49122a110eb19e0378c0f95e156d0f1af0e6dd3d802` |
| DSFormer | `c6689762380369caf9e271f3dc2df905e9a029077c61a59d3483b2910f740ab8` |
| SOFTS | `e4afc67633bfd5644a842289ee5f9729d067698dc51cf1b366c87abdd10afc2d` |
| SparseTSF | `ac274ff7559a2e3d7a6094083b110d13f72e064bc50c5b90684f474d3c3ca038` |
| TimeMixer | `5361ff1753caf43bf3dc0ecd20ac14fa28240f8ee71afd4988e5be1a6bb8bb9a` |
| TimeXer | `248dc51dc46be34f1e24f9c17691adc3681b3545e853b5647f18433094e55a25` |
| UMixer | `743c09622addbe044c0ba59b66f132536f31e261d64e9bfc92a31c092d7faae5` |
| AirFormer | `cfa3d6ba4f29b2201868fcc387f574565a63e12d37a6a8bf8615c7459854db9d` |
| AirPhyNet | `290cb90de6810743a6ed38a97287ef0ee77b95eb545b9b7e641f91c7918ff628` |
| GAGNN | `f49e6f82dedfb86bbb57e79bd3fcc300653d4bae734a082d887b26f524164965` |
| MGSFormer | `64e28204ba06a5c3f058ec2f2d764641620ff38fdaa4dedb7394576b7418887a` |

## Camera-ready baseline mapping status

The path mapping below is source discovery only. For every method, the public package has **not** established a camera-ready-specific command/configuration/data conversion/seed/checkpoint-selection recipe, nor run the baseline. Do not treat the upstream entrypoint hash as proof that the paper's table value can be regenerated.

| Paper baseline | Pinned upstream entrypoint | Camera-ready invocation and result status |
|---|---|---|
| DLinear | `experiments/dlinear/main.py` | Not mapped or executed |
| CATS | `experiments/cats/main.py` | Not mapped or executed |
| CrossGNN | `experiments/crossgnn/main.py` | Not mapped or executed |
| DSFormer | `experiments/dsformer/main.py` | Not mapped or executed |
| SOFTS | `experiments/softs/main.py` | Not mapped or executed |
| SparseTSF | `experiments/sparsetsf/main.py` | Not mapped or executed |
| TimeMixer | `experiments/timemixer/main.py` | Not mapped or executed |
| TimeXer | `experiments/timexer/main.py` | Not mapped or executed |
| UMixer | `experiments/umixer/main.py` | Not mapped or executed |
| AirFormer | `experiments/airformer/main.py` | Not mapped or executed |
| AirPhyNet | `experiments/airphynet/main.py` | Not mapped or executed |
| GAGNN | `experiments/gagnn/main.py` | Not mapped or executed |
| MGSFormer | `experiments/mgsfformer/main.py` | Not mapped or executed |

The three BTH appendix diagnostics reported later in this file are separate archived single-run artifact checks. They do not fill the 13-method main-table mapping gap.

## What remains for exact baseline reproduction

The upstream repository has runnable method entrypoints and a general requirements file, but this release audit has not mapped each camera-ready result to a fully pinned command, per-method configuration, data conversion, seed policy, validation/checkpoint-selection rule, and output metric extraction. The camera-ready paper's dataset is a prepared six-region PM$_{2.5}$ risk-classification setup; it must not be assumed to be interchangeable with CauAir's own default forecasting data layouts. No baseline was run as part of this package validation. A read-only `--help` probe of all 13 entrypoints failed before argument parsing in both available server environments: the AeroMosaic environment lacks `torcheval`; the second available environment additionally lacks `torchdiffeq` for AirPhyNet and `torch_scatter` for GAGNN. This confirms that a separate baseline dependency environment is required; it does not indicate a source-code defect. No packages were installed into the shared environments. Therefore:

- the source provenance is now pinned and the 13 entrypoints are hash-checked;
- no third-party implementation is redistributed by this package;
- the baseline rows in the paper are not claimed to be regenerated by this release;
- a complete baseline reproduction supplement still needs an author-verified method-by-method config/command/data mapping and execution evidence.

Before using the external baseline sources, inspect the upstream repository's license and the licenses of each included implementation/dependency. This note is provenance, not a legal determination.

## Matched BTH no-oracle baseline audit

The archived single-run MGSFormer and TimeMixer diagnostic summaries were also checked against the October 4 paper appendix:

| Parent-information protocol | MGSFormer Micro/Macro F1 | TimeMixer Micro/Macro F1 |
|---|---:|---:|
| No parent | 0.6303 / 0.5091 | 0.5545 / 0.4083 |
| Predicted only | 0.6272 / 0.5070 | 0.5410 / 0.4166 |
| Historical upstream PM2.5 | 0.6318 / 0.5342 | 0.5557 / 0.4052 |

These values were artifact-checked, not freshly rerun. The MGSFormer BTH control configuration records seed 42, 30 epochs, batch/eval-batch 32/64, learning rate `1e-3`, weight decay `1e-4`, dropout `0.2`, hidden width 64, four attention heads, parent cap 12, and weighted loss. The archived TimeMixer results labeled `formal` match the paper; an older server-side `parent_context_timemixer.py` experiment produced substantially different values and must not be substituted for the reported TimeMixer control. The command/config/source path for the exact formal TimeMixer artifact is not fully assembled into this public package, so this audit does not yet meet the reproducibility bar.

The Main Table's 13 baselines and these matched BTH controls are distinct scopes. Pinning the CauAir entrypoints does not establish that each main-table paper result used the same code/config/data transformation or checkpoint-selection protocol. The method-by-method main-table mapping and fresh execution remain outstanding.
