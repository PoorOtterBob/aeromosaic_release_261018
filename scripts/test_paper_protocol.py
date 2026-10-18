"""Data-free checks for the camera-ready input and DAG contract."""
import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from prepare_camera_ready_data import dag, oracle_prompt, precipitation_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.build_e10_noparent_bundle import replace_upstream_block as remove_parent_labels
from diagnostics.build_e7_lag1_pm25_bundle import replace_upstream_block as insert_historical_parent_values


class PaperProtocolTests(unittest.TestCase):
    def test_no_oracle_builders_rewrite_the_distributed_prompt_schema(self):
        source = (
            "Region: BTH. City: Beijing. Target date: 2015-01-02. DAG level: 5. "
            "Upstream parent-label context: Baoding: class 4, Cangzhou: class 1. "
            "Predict the PM2.5 risk class."
        )
        no_parent = remove_parent_labels(source, "unknown/no-parent")
        self.assertIn("Upstream parent-label context: unknown/no-parent.", no_parent)
        self.assertNotIn("class 4", no_parent)
        historical = insert_historical_parent_values(
            source, "Baoding: previous-day observed PM2.5=42.0 ug/m3"
        )
        self.assertIn("previous-day observed PM2.5=42.0", historical)
        self.assertNotIn("class 4", historical)

    def test_precipitation_thresholds_are_left_closed(self):
        examples = ((0.0, "No noticeable"), (0.1, "light rain"),
                    (10.0, "moderate rain"), (25.0, "heavy rain"),
                    (50.0, "rainstorm"), (100.0, "severe rainstorm"),
                    (250.0, "extreme rainstorm"))
        for mm, expected in examples:
            with self.subTest(mm=mm):
                values = np.zeros(8)
                values[0] = mm / 1000.0
                self.assertIn(expected, precipitation_text(values))

    def test_precipitation_requires_original_finite_metres(self):
        for values in (np.full(8, np.nan), np.full(8, -0.1), np.zeros(7)):
            with self.assertRaises(ValueError):
                precipitation_text(values)

    def test_dag_is_binary_destination_by_source(self):
        series = np.zeros((3, 8, 13), dtype=np.float32)
        series[0, -1, 7] = 1.0  # source 0 blows east toward city 1
        xy = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        a, levels, roots = dag(series, xy)
        self.assertTrue(np.all((a == 0) | (a == 1)))
        self.assertEqual(a[1, 0], 1)
        self.assertEqual(a[0, 1], 0)
        self.assertTrue(np.array_equal(roots, (a.sum(axis=1) == 0).astype(np.int64)))
        dst, src = np.nonzero(a)
        self.assertTrue(np.all(levels[dst] > levels[src]))

    def test_oracle_prompt_marks_privileged_parent_labels(self):
        prompt = oracle_prompt("BTH", "Tianjin", "2018-01-01", 1,
                               [("Beijing", 2)], "No noticeable recent precipitation.")
        self.assertIn("oracle protocol: target-date parent labels", prompt)
        self.assertIn("- Beijing: label=2", prompt)
        self.assertIn("Output only one label token: 0, 1, 2, 3, 4, or 5.", prompt)

    def test_validator_accepts_complete_bundle_and_rejects_contract_violations(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            assets = bundle / "assets" / "train"
            assets.mkdir(parents=True)
            for split in ("val", "test"):
                directory = bundle / "assets" / split
                directory.mkdir(parents=True)
                (directory / "group_index.jsonl").write_text("", encoding="utf-8")
            files = {}
            n = 13
            adjacency = np.zeros((n, n), dtype=np.float32)
            adjacency[1, 0] = 1
            arrays = {"raw_ts": np.zeros((n, 8, 13), dtype=np.float32),
                      "labels": np.arange(n, dtype=np.int64) % 6,
                      "dag_adj": adjacency,
                      "dag_level": np.array([0, 1] + [0] * (n - 2), dtype=np.int64),
                      "is_root": np.array([1, 0] + [1] * (n - 2), dtype=np.int64)}
            for name, array in arrays.items():
                path = assets / (name + ".npy")
                np.save(path, array)
                files[name + "_path"] = str(path)
            parent_count = assets / "parent_count.npy"
            np.save(parent_count, (arrays["dag_adj"] > 0).sum(axis=1).astype(np.int64))
            files["parent_count_path"] = str(parent_count)
            cities = assets / "city_names.json"
            city_names = ["Beijing", "Tianjin"] + [f"BTH-{i}" for i in range(2, n)]
            cities.write_text(json.dumps(city_names), encoding="utf-8")
            files["city_names_path"] = str(cities)
            prompts = assets / "prompts.json"
            prompt_rows = [
                {"prompt": oracle_prompt("BTH", "Beijing", "2015-01-02", 0, [],
                                         "No noticeable recent precipitation.")},
                {"prompt": oracle_prompt("BTH", "Tianjin", "2015-01-02", 1,
                                         [("Beijing", 0)], "No noticeable recent precipitation.")}
            ]
            prompt_rows.extend(
                {"prompt": oracle_prompt("BTH", city_names[i], "2015-01-02",
                                         int(arrays["labels"][i]), [],
                                         "No noticeable recent precipitation.")}
                for i in range(2, n)
            )
            prompts.write_text(json.dumps(prompt_rows), encoding="utf-8")
            files["prompts_path"] = str(prompts)
            (assets / "group_index.jsonl").write_text(json.dumps(files) + "\n", encoding="utf-8")
            manifest = {"region": "BTH", "dag_wind": "last observation in historical window, channels 7-8",
                        "dag": {"adjacency": "binary destination-by-source"},
                        "precipitation_text": "raw metres, previous eight steps"}
            (bundle / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).with_name("validate_data_contract.py")),
                       "--bundle", str(bundle), "--region", "BTH", "--paper-protocol"]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            index_path = assets / "group_index.jsonl"
            complete_row = json.loads(index_path.read_text(encoding="utf-8"))
            incomplete_row = dict(complete_row)
            incomplete_row.pop("parent_count_path")
            index_path.write_text(json.dumps(incomplete_row) + "\n", encoding="utf-8")
            missing_parent_count = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(missing_parent_count.returncode, 0)
            self.assertIn("parent_count_path", missing_parent_count.stdout)
            index_path.write_text(json.dumps(complete_row) + "\n", encoding="utf-8")
            arrays["dag_adj"][1, 0] = 0.5
            np.save(files["dag_adj_path"], arrays["dag_adj"])
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("binary", failed.stdout)

            arrays["dag_adj"][1, 0] = 1
            np.save(files["dag_adj_path"], arrays["dag_adj"])
            invalid_labels = arrays["labels"].copy()
            invalid_labels[0] = 99
            np.save(files["labels_path"], invalid_labels)
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("0 through 5", failed.stdout)


if __name__ == "__main__":
    unittest.main()
