#!/usr/bin/env python3
"""Unit, statistical, alignment, and artifact parity tests for Experiment 07."""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import analysis as A  # noqa: E402


class TestPrimitives(unittest.TestCase):
    def test_parse_bounded_float_rejects_bad_inputs(self):
        self.assertAlmostEqual(A.parse_bounded_float(0.5, 0.0, 1.0, "ok"), 0.5)
        for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                A.parse_bounded_float(bad, 0.0, 1.0, "bad")

    def test_proper_scores_and_decomposition(self):
        q = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
        ratings = [3, 4, 4]
        rps, dist_err, disag = A.rps_decomp(q, ratings)
        self.assertAlmostEqual(rps, dist_err + disag, places=12)
        self.assertGreaterEqual(dist_err, 0.0)
        self.assertGreaterEqual(disag, 0.0)
        self.assertGreater(A.ce_item(q, ratings), 0.0)
        self.assertGreaterEqual(A.tvd_item(q, ratings), 0.0)
        self.assertLessEqual(A.tvd_item(q, ratings), 1.0)
        self.assertAlmostEqual(A.brier_item(q, [3]), float(np.sum(q * q) - 2.0 * q[3] + 1.0), places=12)

    def test_calibrated_repeat_tvd(self):
        scores = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
        ratings = [[0, 1], [1, 1], [2, 3], [4, 4], [5, 5], [5, 5]]
        cal = A.fit_ce_ordinal(scores, ratings)
        rows = [{"kind": "crit", "item": "it_same", "task": "t1", "level": 0},
                {"kind": "crit", "item": "it_diff", "task": "t2", "level": 0}]
        raw = {("crit", "it_same", 0): [0.8, 0.8, 0.8], ("crit", "it_diff", 0): [0.2, 0.8, 0.9]}
        out = A.eval_repeat_stats(rows, raw, cal_fn=cal, min_repeats=2)
        self.assertTrue(out["computable"])
        self.assertGreater(out["v_rep"], 0.0)
        self.assertGreater(out["cal_tvd"], 0.0)
        self.assertGreater(out["cal_ev_range_05"], 0.0)

    def test_nonneg_ridge_ati_nonnegative_and_no_leakage(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0.2, 1.0, size=(24, 6))
        true_w = np.array([0.4, 0.3, 0.0, 0.2, 0.0, 0.1])
        y = -0.25 + X @ true_w + rng.normal(0.0, 0.01, size=24)
        fams = ["f1"] * 6 + ["f2"] * 6 + ["f3"] * 6 + ["f4"] * 6
        w, b, lam = A.fit_nonneg_ridge_ati(X, y, fams)
        self.assertTrue(np.all(w >= -1e-12))
        self.assertIn(lam, A.RIDGE_LAMBDAS)
        preds = A.predict_nonneg_ridge(X, w, b)
        self.assertGreater(A.spearman_corr(preds, y), 0.90)

    def test_unequal_cluster_bootstrap_deterministic_and_ordered(self):
        pairs = [("t_big", 0.0)] * 8 + [("t_big", 1.0)] * 2 + [("t_small", 0.9), ("t_mid", 0.3), ("t_mid", 0.5)]
        b1 = A.cluster_mean_ci(pairs, draws=300, seed=A.SEED)
        b2 = A.cluster_mean_ci(pairs, draws=300, seed=A.SEED)
        self.assertEqual(b1, b2)
        self.assertLessEqual(b1["ci_lo"], b1["value"])
        self.assertLessEqual(b1["value"], b1["ci_hi"])
        vals_a = [v for _, v in pairs]
        vals_b = [0.4] * len(pairs)
        tids = [t for t, _ in pairs]
        sf = A.paired_task_signflip_p(tids, vals_a, vals_b, draws=200, seed=A.SEED)
        ci = A.cluster_mean_ci([(t, va - vb) for t, va, vb in zip(tids, vals_a, vals_b)], draws=200, seed=A.SEED)
        self.assertAlmostEqual(sf["obs_diff"], ci["value"], places=12)


class TestLoggedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.records, cls.part_b_rows = A.build_summary()

    def test_alignment_gates_and_verdicts(self):
        al = self.summary["alignment"]
        self.assertGreaterEqual(al["gate_a1_e68_checks"], 400)
        self.assertLessEqual(al["gate_a1_e68_max_abs_diff"], 1e-9)
        self.assertGreaterEqual(al["gate_a2_exp03_checks"], 450)
        self.assertLessEqual(al["gate_a2_exp03_max_abs_diff"], 1e-9)
        self.assertEqual(self.summary["new_model_calls"], 0)
        self.assertEqual(self.summary["cost_usd"], 0.0)
        self.assertEqual(self.summary["part_a"]["verdict_h1_proper"], "supported")
        self.assertEqual(self.summary["part_b"]["verdict_h_idea3"], "supported")
        self.assertEqual(self.summary["part_c"]["verdict_h2_new"], "supported")
        pb = self.summary["part_b"]
        self.assertEqual(pb["verdict_h_idea3_prereg_literal"], "supported")
        self.assertEqual(pb["verdict_h_idea3_leak_free_prereg"], "falsified")
        self.assertEqual(pb["verdict_h_idea3_leak_free_extension_ati_min"], "supported")
        self.assertEqual(len(pb["lomfo_folds"]), 6)
        self.assertIn("lomfo_fold_summary", pb)
        for row in self.part_b_rows:
            self.assertIn("ctl_rejected_trainval_q_of_4", row)
            self.assertIn("q_const_tv_median_index", row)
            if row["split"] == "lomfo_oof_all" and row["index"] == "ATI_cal_rps":
                self.assertEqual(row["inference_mode"], "descriptive_only_pooled_oof")
                self.assertIsNone(row["rps_p"])
                self.assertIsNone(row["rps_ci_lo"])
                self.assertIsNone(row["rps_within_fam_p"])
            else:
                self.assertEqual(row["inference_mode"], "valid_fixed_predictor")
                self.assertIsNotNone(row["ce_ci_lo"])
                self.assertIsNotNone(row["tvd_within_fam_p"])
        for ds in self.summary["part_c"]["datasets"].values():
            r5 = ds["rating_weighted_p_eq_5"]
            self.assertLessEqual(r5["ci_lo"], r5["value"])
            self.assertLessEqual(r5["value"], r5["ci_hi"])

    def test_ci_ordering_recursive(self):
        def walk(obj, path="root"):
            if isinstance(obj, dict):
                if {"ci_lo", "ci_hi"}.issubset(obj.keys()):
                    val = obj.get("value", obj.get("spearman"))
                    lo, hi = obj["ci_lo"], obj["ci_hi"]
                    if val is not None and lo is not None and hi is not None:
                        self.assertLessEqual(lo, val + 1e-12, path)
                        self.assertLessEqual(val, hi + 1e-12, path)
                for k, v in obj.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")

        walk(self.summary)

    def test_saved_artifacts_parity_and_privacy(self):
        res_dir = EXP / "results"
        stored = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
        json.dumps(stored, allow_nan=False)
        self.assertEqual(stored["part_a"], self.summary["part_a"])
        self.assertEqual(stored["part_b"], self.summary["part_b"])
        self.assertEqual(stored["part_c"], self.summary["part_c"])
        self.assertEqual(stored["alignment"], self.summary["alignment"])

        with tempfile.TemporaryDirectory() as tmp:
            A.write_outputs(self.summary, self.records, self.part_b_rows, Path(tmp))
            for fn in (
                "part_a_proper_scores_vs_properties.csv",
                "part_b_idea3_certification.csv",
                "part_c_human_regimes.csv",
                "canonical_lines.txt",
            ):
                self.assertEqual(
                    (Path(tmp) / fn).read_text(encoding="utf-8"),
                    (res_dir / fn).read_text(encoding="utf-8"),
                    fn,
                )

        with (res_dir / "part_a_proper_scores_vs_properties.csv").open(encoding="utf-8") as f:
            rows_a = list(csv.DictReader(f))
        self.assertEqual(len(rows_a), 60)

        for fig in (
            "fig1_proper_scores_vs_axiomatic_violations.png",
            "fig2_idea3_out_of_family_certification.png",
            "fig3_h2_human_ceiling_vs_disagreement.png",
        ):
            self.assertEqual((res_dir / fig).read_bytes()[:8], b"\x89PNG\r\n\x1a\n", fig)

        blob = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in res_dir.glob("*")
            if p.suffix != ".png"
        )
        for banned in ("api_key", "sk-", "BEGIN PRIVATE", "BEGIN OPENSSH", "password", "participant1_data"):
            self.assertNotIn(banned, blob)

    def test_results_md_quotes_canonical_lines_and_terms(self):
        import re

        results_md = EXP / "RESULTS.md"
        if not results_md.exists():
            self.skipTest("RESULTS.md not yet written")
        body = results_md.read_text(encoding="utf-8")
        lines = (EXP / "results" / "canonical_lines.txt").read_text(encoding="utf-8").splitlines()
        missing = [ln for ln in lines if ln and ln not in body]
        self.assertEqual(missing, [])
        for md_path in (EXP / "README.md", results_md):
            md_txt = md_path.read_text(encoding="utf-8")
            self.assertNotIn("github.com/brandomiranda/", md_txt, str(md_path))
            for rel in re.findall(r"\[[^\]]+\]\(([^)#]+)\)", md_txt):
                if rel.startswith(("http://", "https://", "file://", "mailto:")):
                    continue
                target = (md_path.parent / rel).resolve()
                self.assertTrue(target.exists(), f"Broken relative link {rel} in {md_path}")


if __name__ == "__main__":
    unittest.main()
