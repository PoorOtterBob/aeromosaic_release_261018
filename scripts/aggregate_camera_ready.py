#!/usr/bin/env python3
"""Aggregate the exact six-region, three-seed abl_AC main-table summaries."""
import argparse
import json
import statistics
from pathlib import Path

REGIONS = ("BTH", "YRD", "SDP", "ZHD", "CJD", "CYB")
SEEDS = (42, 123, 456)
METRICS = ("accuracy", "micro_f1", "macro_f1")
PAPER_MEANS = {
    "BTH": {"micro_f1": 0.8387, "macro_f1": 0.7239},
    "YRD": {"micro_f1": 0.7671, "macro_f1": 0.5976},
    "SDP": {"micro_f1": 0.8166, "macro_f1": 0.6218},
    "ZHD": {"micro_f1": 0.7861, "macro_f1": 0.6466},
    "CJD": {"micro_f1": 0.8674, "macro_f1": 0.5499},
    "CYB": {"micro_f1": 0.8551, "macro_f1": 0.6802},
}
PAPER_POPULATION_SDS = {
    "BTH": {"micro_f1": 0.0061, "macro_f1": 0.0292},
    "YRD": {"micro_f1": 0.0035, "macro_f1": 0.0033},
    "SDP": {"micro_f1": 0.0025, "macro_f1": 0.0084},
    "ZHD": {"micro_f1": 0.0064, "macro_f1": 0.0154},
    "CJD": {"micro_f1": 0.0053, "macro_f1": 0.0331},
    "CYB": {"micro_f1": 0.0088, "macro_f1": 0.0096},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="output root containing the abl_AC directory, or abl_AC itself")
    parser.add_argument("--out", type=Path, default=Path("camera_ready_aggregate.json"))
    args = parser.parse_args()
    runs_root = args.runs_root
    if (runs_root / "abl_AC").is_dir():
        runs_root = runs_root / "abl_AC"
    results = {}
    errors = []
    for region in REGIONS:
        rows = []
        for seed in SEEDS:
            run_dir = runs_root / f"{region}_seed{seed}_eval_test"
            path = run_dir / "test" / "summary.json"
            archived_path = run_dir / "summary_all.json"
            if not path.is_file() and archived_path.is_file():
                path = archived_path
            if not path.is_file():
                errors.append(
                    f"missing {region} seed={seed}: expected "
                    f"{run_dir / 'test' / 'summary.json'} or {archived_path}"
                )
                continue
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                if "summaries" in summary:
                    matches = [row for row in summary["summaries"] if row.get("split") == "test"]
                    if len(matches) != 1:
                        raise ValueError("summary_all.json must contain exactly one test split")
                    summary = matches[0]
                rows.append({"seed": seed, **{metric: float(summary[metric]) for metric in METRICS}})
            except Exception as exc:
                errors.append(f"invalid {path}: {exc}")
        if len(rows) == len(SEEDS):
            results[region] = {
                "seeds": rows,
                **{metric: {"mean": statistics.mean(r[metric] for r in rows),
                            "population_sd": statistics.pstdev(r[metric] for r in rows)}
                   for metric in METRICS},
            }
    if errors:
        print("FAIL: all expected paper test summaries are required.")
        for error in errors:
            print(f"- {error}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "regions": results,
        "regions_order": list(REGIONS),
        "seeds": list(SEEDS),
        "std_definition": "population standard deviation across the three archived seeds (ddof=0)",
        "source": "abl_AC/<REGION>_seed<SEED>_eval_test/test/summary.json (release output) or sibling summary_all.json (archived output)",
        "paper_mean_check": {
            region: {
                metric: round(results[region][metric]["mean"], 4) == expected
                for metric, expected in PAPER_MEANS[region].items()
            }
            for region in REGIONS
        },
        "paper_population_sd_check": {
            region: {
                metric: round(results[region][metric]["population_sd"], 4) == expected
                for metric, expected in PAPER_POPULATION_SDS[region].items()
            }
            for region in REGIONS
        },
    }, indent=2) + "\n", encoding="utf-8")
    failed = [
        f"{region}/{metric}"
        for region in REGIONS
        for metric, expected in PAPER_MEANS[region].items()
        if round(results[region][metric]["mean"], 4) != expected
    ]
    if failed:
        print(f"FAIL: rounded means differ from the paper main table: {', '.join(failed)}")
        return 1
    failed_sds = [
        f"{region}/{metric}"
        for region in REGIONS
        for metric, expected in PAPER_POPULATION_SDS[region].items()
        if round(results[region][metric]["population_sd"], 4) != expected
    ]
    if failed_sds:
        print(f"FAIL: rounded population SDs differ from the paper main table: {', '.join(failed_sds)}")
        return 1
    print(f"PASS: all 12 rounded means and 12 population SDs match; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
