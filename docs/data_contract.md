# Group-bundle data contract

The six main-table experiment bundles are not redistributed. Each region path in `configs/paths.yaml` points to a prepared region bundle:

```text
<region_bundle>/
  assets/
    train/group_index.jsonl
    val/group_index.jsonl
    test/group_index.jsonl
```

Each non-empty JSONL record must include `raw_ts_path`, `labels_path`, `dag_adj_path`, `dag_level_path`, `is_root_path`, `parent_count_path`, `city_names_path`, and `prompts_path`. Paths may be absolute, relative to the bundle root, or relative to the experiment project root. The supplied runner sets its working directory to `project_root`, so project-root-relative paths resolve there. The validator supports the same case via `--project-root`. The current builder emits all required fields; `scripts/add_parent_count_to_bundle.py` is retained only as a migration utility for older bundles.

For main-table reproduction, use the six frozen experiment-time precipitation bundles named in `configs/paths.example.yaml`. They predate the later unified manifest and are validated with the generic contract below. The optional `--paper-protocol` mode applies only to newly reconstructed binary/last-historical bundles; the main-table runner deliberately does not require it.

For N city nodes, the primary arrays must have these shapes:

- `raw_ts`: (N, 8, 13), float32-compatible
- `labels`: (N,), integer class IDs 0 through 5
- `dag_adj`: (N, N), float32-compatible; weighted for BTH and binary for the other five frozen regional bundles
- `dag_level`: (N,)
- `is_root`: (N,)
- `parent_count`: (N,), equal to the number of positive incoming adjacency entries
- `city_names`: JSON list of N city names

The stored raw tensor has 13 channels. For model-visible numerical prefix tokens, root-region fusion is applied first and then only channels `{0,11,12}` are kept, so meteorological columns 1--10 do not enter the numerical prefix. Precipitation text and oracle parent-label context are read from the frozen prompt payloads. Prompt payloads must be a JSON list with one string or `{"prompt": "..."}` object per city.

Validate a bundle before training:

```bash
python scripts/validate_data_contract.py --bundle /path/to/region_bundle --project-root /path/to/aeromosaic --region BTH
```

The validator reads local files without copying or modifying them. Generic mode checks every trainer/evaluator-required indexed path; the expected node count for the selected region; array shapes; numeric finiteness; integer labels in the range 0--5; non-negative, loop-free adjacency; integer roots, levels and parent counts; unique non-empty city names; and prompt count. Optional paper-protocol mode additionally checks the later reconstruction manifest, binary adjacency, level ordering, root mask and oracle prompt fields. Neither mode independently proves raw-source provenance.

## Bundled BTH no-oracle contract

`data/bth_no_oracle_source_8x14` is a separate, self-contained prepared input for the matched BTH no-oracle diagnostics. It is not one of the six main-table bundles. It uses the same JSONL field names and 13 BTH cities, but `raw_ts` has shape `(13, 8, 14)`. The bundle contains 730/365/365 daily train/validation/test groups; every indexed path is release-root-relative and remains inside the bundle. Check it with:

```bash
python scripts/validate_bth_no_oracle_bundle.py --release-root .
```

The accompanying `bundle_manifest.json` records the source-index hashes and the provenance boundary: this is the preserved experiment-time prepared input, not a converter from public raw KnowAir data.
