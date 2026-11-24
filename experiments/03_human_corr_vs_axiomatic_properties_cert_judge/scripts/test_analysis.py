#!/usr/bin/env python3
"""Unit and integration tests for Experiment 03 analysis.py (2026-11-24)."""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

import analysis as A  # noqa: E402


class TestAnalysisPrimitives(unittest.TestCase):
    def test_perfect_and_constant_judges(self) -> None:
        rows = [
            {"task": "t1", "kind": "bm_gold", "level": 1},
            {"task": "t2", "kind": "bm_gold", "level": 1},
            {"task": "t2", "kind": "bm_gold", "level": 2},
            {"task": "t3", "kind": "bm_gold", "level": 1},
            {"task": "t4", "kind": "bm_gold", "level": 1},
        ]
        perfect_scores = {
            ("ident", "t1", 0): 1.0,
            ("ident", "t2", 0): 1.0,
            ("ident", "t3", 0): 0.8,
            ("ident", "t4", 0): 1.0,
            ("bm_gold", "t1", 1): 0.6,
            ("bm_gold", "t2", 1): 0.8,
            ("bm_gold", "t2", 2): 0.4,
            ("bm_gold", "t3", 1): 0.6,
            ("bm_gold", "t4", 1): 0.2,
        }
        res_perf = A.summarize_paired_violation(
            rows,
            perfect_scores,
            base_lookup=lambda r: ("ident", r["task"], 0),
            probe_lookup=lambda r: ("bm_gold", r["task"], r["level"]),
            mode="ge_base",
        )
        self.assertTrue(res_perf["computable"])
        self.assertAlmostEqual(res_perf["value"], 0.0)
        self.assertAlmostEqual(res_perf["strict_gt"], 0.0)
        self.assertAlmostEqual(res_perf["tie_eq"], 0.0)

        const_scores = {k: 1.0 for k in perfect_scores}
        res_const = A.summarize_paired_violation(
            rows,
            const_scores,
            base_lookup=lambda r: ("ident", r["task"], 0),
            probe_lookup=lambda r: ("bm_gold", r["task"], r["level"]),
            mode="ge_base",
        )
        self.assertTrue(res_const["computable"])
        self.assertAlmostEqual(res_const["value"], 1.0)
        self.assertAlmostEqual(res_const["strict_gt"], 0.0)
        self.assertAlmostEqual(res_const["tie_eq"], 1.0)

    def test_coverage_gating_and_abs_diff(self) -> None:
        rows = [{"task": f"t{i}", "kind": "fmt_layout", "level": 1} for i in range(5)]
        scores = {
            ("ident", "t0", 1): 1.0,
            ("ident", "t1", 1): 1.0,
            ("ident", "t2", 1): 1.0,
            ("fmt_layout", "t0", 1): 1.0,
            ("fmt_layout", "t1", 1): 0.6,
            ("fmt_layout", "t2", 1): 1.0,
        }
        # Only 3/5 = 60% < 80% threshold
        res_low = A.summarize_paired_violation(
            rows,
            scores,
            base_lookup=lambda r: ("ident", r["task"], 1),
            probe_lookup=lambda r: ("fmt_layout", r["task"], 1),
            mode="abs_diff",
        )
        self.assertFalse(res_low["computable"])
        self.assertIsNone(res_low["value"])
        self.assertAlmostEqual(res_low["coverage"], 0.6)

        # 4/5 = 80% >= 80% threshold
        scores[("ident", "t3", 1)] = 1.0
        scores[("fmt_layout", "t3", 1)] = 1.0
        res_ok = A.summarize_paired_violation(
            rows,
            scores,
            base_lookup=lambda r: ("ident", r["task"], 1),
            probe_lookup=lambda r: ("fmt_layout", r["task"], 1),
            mode="abs_diff",
        )
        self.assertTrue(res_ok["computable"])
        self.assertAlmostEqual(res_ok["value"], 0.25)
        self.assertAlmostEqual(res_ok["mean_abs_gap"], 0.1)

    def test_repeat_inconsistency_and_bootstrap_reproducibility(self) -> None:
        rows = [
            {"kind": "crit", "key": "t1__a0", "task": "t1", "level": 0},
            {"kind": "crit", "key": "t2__a0", "task": "t2", "level": 0},
        ]
        raw = {
            ("crit", "t1__a0", 0): [0.8, 0.8, 0.8],
            ("crit", "t2__a0", 0): [0.2, 0.8, 0.8],
        }
        rep = A.summarize_repeat_inconsistency(rows, raw, min_repeats=2)
        self.assertTrue(rep["computable"])
        self.assertAlmostEqual(rep["value"], 0.5)
        self.assertAlmostEqual(rep["mean_range"], 0.3)
        expected_sd2 = math.sqrt(((0.2 - 0.6) ** 2 + 2 * (0.8 - 0.6) ** 2) / 3)
        self.assertAlmostEqual(rep["mean_sd_pop"], expected_sd2 / 2.0)

        c1 = A.cluster_mean_ci([("t1", 0.0), ("t2", 1.0), ("t3", 0.5)], draws=500, seed=A.SEED)
        c2 = A.cluster_mean_ci([("t1", 0.0), ("t2", 1.0), ("t3", 0.5)], draws=500, seed=A.SEED)
        self.assertEqual(c1, c2)

    def test_synthetic_human_consistency_no_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hdir = Path(tmp)
            (hdir / "old_prompt").mkdir()
            items = [
                ("task1.lean", "task1__agent0.lean"),
                ("task2.lean", "task2__agent0.lean"),
                ("task3.lean", "task3__agent0.lean"),
            ]
            data = {
                "participant1_data_correct_prompt.tsv": [5, 3, 1],
                "participant2_data_pass_1.tsv": [5, 1, 3],
                "participant3_data_pass_1.tsv": [5, "", 4],
                "participant4_data_pass_1.tsv": [5, 2, 1],
                "participant5_data_pass_1.tsv": [5, 4, 2],
                "old_prompt/participant1_data_pass_1.tsv": [5, 4, 2],
                "old_prompt/participant1_data_pass_2.tsv": [4, 2, 2],
            }
            for rel, scores in data.items():
                p = hdir / rel
                with p.open("w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["gold_file", "candidate_file", "human_score_0_5"])
                    for (g, c), s in zip(items, scores):
                        w.writerow([g, c, s])

            res = A.analyze_humans(hdir)
            self.assertEqual(res["rater_counts"]["P3"], 2)
            self.assertEqual(res["fixed_pool_pairs"]["P1_correct__vs__P3"]["n"], 2)
            # Item 1 has range 0; item 2 has {3,1,2,4} -> range 3; item 3 has {1,3,4,1,2} -> range 3
            self.assertAlmostEqual(res["item_spread_all_ge2"]["range_ge_2"]["value"], 2 / 3)
            self.assertEqual(res["item_spread_complete5"]["n_items"], 2)


class TestGeneratedArtifacts(unittest.TestCase):
    def test_outputs_alignment_and_privacy(self) -> None:
        summary_path = RESULTS_DIR / "summary.json"
        human_path = RESULTS_DIR / "human_consistency.json"
        csv_path = RESULTS_DIR / "judge_table.csv"
        self.assertTrue(summary_path.exists(), "Run scripts/analysis.py before running integration tests.")
        self.assertTrue(human_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue((RESULTS_DIR / "fig1_human_corr_vs_property_violations.png").stat().st_size > 1000)
        self.assertTrue((RESULTS_DIR / "fig2_human_inconsistency_lean4.png").stat().st_size > 1000)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        human = json.loads(human_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["alignment_check"]["passed"])
        self.assertLessEqual(summary["alignment_check"]["max_abs_diff"], 1e-9)
        self.assertGreaterEqual(summary["alignment_check"]["checks"], 200)
        self.assertEqual(summary["n_units"], 60)
        self.assertEqual(summary["n_llm_units"], 56)
        self.assertEqual(summary["n_control_units"], 4)
        self.assertIn("n_failing_high_corr_excluding_v_bug", summary)
        self.assertEqual(summary["n_failing_high_corr_excluding_v_bug"], 22)
        self.assertEqual(human["n_pool_tasks"], 25)
        self.assertEqual(human["item_spread_all_ge2"]["n_tasks"], 25)
        self.assertEqual(human["within_human_shifts"]["P1_old_pass1__vs__P1_old_pass2"]["n_tasks"], 25)

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 60)
        self.assertIn("rho_heldout_n_items", csv_rows[0])
        self.assertIn("rho_heldout_n_declared", csv_rows[0])
        for u in summary["units"]:
            self.assertIn("rho_heldout_n_items", u)
            self.assertIn("rho_heldout_n_declared", u)

        # Privacy check: never leak raw Lean filenames, candidate IDs, or annotator names
        for path in (summary_path, human_path, csv_path):
            txt = path.read_text(encoding="utf-8")
            self.assertNotIn(".lean", txt)
            self.assertNotIn("__agent", txt)
            self.assertNotIn("Ethan", txt)
            self.assertNotIn("ethan", txt.lower())


if __name__ == "__main__":
    unittest.main()
