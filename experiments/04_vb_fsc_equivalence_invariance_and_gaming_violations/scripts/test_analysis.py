#!/usr/bin/env python3
"""Unit and integration tests for Experiment 04 analysis.py (2026-11-24)."""
from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
sys.path.insert(0, str(SCRIPTS_DIR))

import analysis as A  # noqa: E402


class TestExp04Primitives(unittest.TestCase):
    def test_synthetic_jra_equivalence_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jdir = Path(tmp)
            ab_fields = [
                "judge_id", "tier", "harness", "model", "effort", "method",
                "task_id", "module_id", "truth", "stratum", "condition", "repeat_idx",
                "parsed_score_0_10", "fsc_norm", "harsh_credit_ge9", "harsh_credit_eq10",
                "cost_usd", "input_tokens", "output_tokens", "error",
            ]
            ab_rows = [
                # Anchored M1 judge: match=1.0, diff=0.5 on t1 and t2
                ("j1", "frontier", "h1", "m1", "default", "M1_deployed_tc_v1", "t1", "id1", "T1", "s", "match_refactored", "0", "10", "1.0", "1", "1", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "default", "M1_deployed_tc_v1", "t1", "id2", "T1", "s", "diff_gold", "0", "5", "0.5", "0", "0", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "default", "M1_deployed_tc_v1", "t2", "id3", "T1", "s", "match_refactored", "0", "10", "1.0", "1", "1", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "default", "M1_deployed_tc_v1", "t2", "id4", "T1", "s", "diff_gold", "0", "5", "0.5", "0", "0", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "default", "M1_deployed_tc_v1", "t3", "id5", "T0", "s", "non_covering", "0", "6", "0.6", "0", "0", "0", "1", "1", ""),
                # Order-violating M4 judge: dual_ab=0.9, dual_ba=0.6
                ("j1", "frontier", "h1", "m1", "default", "M4_dual_ref", "t1", "id6", "T1", "s", "dual_ab", "0", "9", "0.9", "1", "0", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "default", "M4_dual_ref", "t1", "id7", "T1", "s", "dual_ba", "0", "6", "0.6", "0", "0", "0", "1", "1", ""),
                # Mixed-effort judge M5: t1 matches effort (diff=0), t2 mismatches effort (shift=0.3)
                ("j1", "frontier", "h1", "m1", "medium", "M5_harness_ref_hidden_lean", "t1", "id8", "T1", "s", "match_refactored", "0", "10", "1.0", "1", "1", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "medium", "M5_harness_ref_hidden_lean", "t1", "id9", "T1", "s", "diff_gold", "0", "10", "1.0", "1", "1", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "medium", "M5_harness_ref_hidden_lean", "t2", "id10", "T1", "s", "match_refactored", "0", "7", "0.7", "0", "0", "0", "1", "1", ""),
                ("j1", "frontier", "h1", "m1", "low", "M5_harness_ref_hidden_lean", "t2", "id11", "T1", "s", "diff_gold", "0", "4", "0.4", "0", "0", "0", "1", "1", ""),
            ]
            with (jdir / "exp_a_b_per_item_scores.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(ab_fields)
                w.writerows(ab_rows)
            with (jdir / "exp_b_cli_agent_scores.csv").open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(ab_fields)

            ec_fields = [
                "judge_id", "method", "task_id", "split", "solver_id", "solver_harness", "solver_model",
                "ic_tests", "ic_proofs", "dt_dmean", "dt_dall", "fsc_R_A", "fsc_R_B", "fsc_Dual",
                "harsh_nodt_R_A", "harsh_nodt_R_B", "harsh_nodt_ge09_R_A", "harsh_nodt_ge09_R_B",
                "harsh_dt_R_A", "harsh_dt_R_B",
            ]
            with (jdir / "exp_c_veribench_impact_per_task.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(ec_fields)
                w.writerow(["j1", "M1_deployed_tc_v1", "t1", "easy", "s1", "sh", "sm", "1", "1", "0", "0", "1.0", "0.6", "", "1", "0", "1", "0", "1", "0"])

            sm = {
                "exp_ab_matrix": {
                    "configurations": {
                        "j1::M1_deployed_tc_v1": {
                            "judge_id": "j1",
                            "method": "M1_deployed_tc_v1",
                            "cm": {"mean": 1.0},
                            "cd": {"mean": 0.5},
                            "A_G": {"mean": 0.5},
                            "fc": {"mean": 0.6},
                        }
                    }
                },
                "exp_c_veribench_impact": {
                    "by_judge_method": {"j1::M1_deployed_tc_v1": {"mean_abs_fsc_shift": {"mean": 0.4}}}
                },
            }
            A.write_json(jdir / "summary_metrics.json", sm)

            res = A.analyze_jra(jdir)
            self.assertTrue(res["alignment_jra"]["passed"])
            self.assertEqual(res["h1_equiv_verdict"], "SUPPORTED")
            by_id = {r["config_id"]: r for r in res["part_a_configs"]}
            self.assertAlmostEqual(by_id["j1::M1_deployed_tc_v1"]["V_equiv"]["value"], 1.0)
            self.assertAlmostEqual(by_id["j1::M1_deployed_tc_v1"]["A_G"]["value"], 0.5)
            self.assertAlmostEqual(by_id["j1::M4_dual_ref"]["V_order"]["value"], 1.0)
            m5 = by_id["j1::M5_harness_ref_hidden_lean"]
            self.assertEqual(m5["effort"], "mixed (low/medium)")
            self.assertEqual(m5["n_mismatched_equiv_effort"], 1)
            self.assertAlmostEqual(m5["V_equiv"]["value"], 0.5)
            self.assertAlmostEqual(m5["V_equiv_same_effort"]["value"], 0.0)

            # Heterogeneous model inside one config raises ValueError
            with (jdir / "exp_a_b_per_item_scores.csv").open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(
                    ("j1", "frontier", "h1", "m_other", "low", "M1_deployed_tc_v1", "t9", "id99", "T1", "s", "diff_gold", "0", "5", "0.5", "0", "0", "0", "1", "1", "")
                )
            with self.assertRaises(ValueError):
                A.analyze_jra(jdir)

    def test_synthetic_e23_gaming_and_sign_test(self) -> None:
        jrows = [
            {"task": "t1", "condition": "faithful_gold_copy", "judge_score": "5"},
            {"task": "t2", "condition": "faithful_gold_copy", "judge_score": "5"},
            {"task": "t3", "condition": "faithful_gold_copy", "judge_score": "5"},
            {"task": "t1", "condition": "sorry_proofs", "judge_score": "5"},
            {"task": "t2", "condition": "sorry_proofs", "judge_score": "5"},
            {"task": "t3", "condition": "sorry_proofs", "judge_score": "4"},
            {"task": "t1", "condition": "weakened_props", "judge_score": "1"},
            {"task": "t2", "condition": "weakened_props", "judge_score": "2"},
            {"task": "t3", "condition": "weakened_props", "judge_score": "1"},
        ]
        grows = [
            {"task": "t1", "condition": "faithful_gold_copy", "passes_gate": "1", "passes_gate_and_format": "1"},
            {"task": "t2", "condition": "faithful_gold_copy", "passes_gate": "1", "passes_gate_and_format": "1"},
            {"task": "t3", "condition": "faithful_gold_copy", "passes_gate": "1", "passes_gate_and_format": "1"},
            {"task": "t1", "condition": "sorry_proofs", "passes_gate": "0", "passes_gate_and_format": "0"},
            {"task": "t2", "condition": "sorry_proofs", "passes_gate": "0", "passes_gate_and_format": "0"},
            {"task": "t3", "condition": "sorry_proofs", "passes_gate": "0", "passes_gate_and_format": "0"},
            {"task": "t1", "condition": "weakened_props", "passes_gate": "1", "passes_gate_and_format": "1"},
            {"task": "t2", "condition": "weakened_props", "passes_gate": "1", "passes_gate_and_format": "1"},
            {"task": "t3", "condition": "weakened_props", "passes_gate": "1", "passes_gate_and_format": "1"},
        ]
        recs = A._score_e23_dataset(
            "toy",
            jrows,
            grows,
            [("judge_only", None), ("gated_format", "passes_gate_and_format")],
        )
        m = {(r["condition"], r["pipeline"]): r for r in recs}
        self.assertAlmostEqual(m[("sorry_proofs", "judge_only")]["V_mono"]["value"], 2 / 3)
        self.assertAlmostEqual(m[("sorry_proofs", "judge_only")]["high_credit_ge3"]["value"], 1.0)
        self.assertAlmostEqual(m[("sorry_proofs", "gated_format")]["V_mono"]["value"], 0.0)
        self.assertAlmostEqual(m[("weakened_props", "judge_only")]["mean_drop"]["value"], 11 / 3)
        self.assertAlmostEqual(m[("weakened_props", "judge_only")]["sign_p_val"], 0.25)

        c1 = A.cluster_mean_ci([("t1", 0.0), ("t2", 1.0), ("t3", 0.5)], draws=400, seed=A.SEED)
        c2 = A.cluster_mean_ci([("t1", 0.0), ("t2", 1.0), ("t3", 0.5)], draws=400, seed=A.SEED)
        self.assertEqual(c1, c2)

    def test_unequal_cluster_bootstrap_pvals_and_validation(self) -> None:
        # Unequal clusters: 10 rows in t1 (0.0) vs 1 row in t2 (1.0).
        # Cluster bootstrap draws {t2, t2} with probability 0.25 -> ci_hi must reach 1.0
        # (whereas row-level bootstrap has P(mean==1.0) = (1/11)^11 ~ 0).
        unequal = [("t1", 0.0)] * 10 + [("t2", 1.0)]
        res_u = A.cluster_mean_ci(unequal, draws=600, seed=A.SEED)
        self.assertAlmostEqual(res_u["value"], 1 / 11)
        self.assertAlmostEqual(res_u["ci_hi"], 1.0)

        self.assertIsNone(A.paired_signflip_pvalue([]))
        self.assertAlmostEqual(A.paired_signflip_pvalue([1.0, 1.0, 1.0, 1.0]), 0.125)
        self.assertIsNone(A.exact_sign_test_p([]))
        self.assertAlmostEqual(A.exact_sign_test_p([0.0, 0.0, 0.0]), 1.0)

        with self.assertRaises(ValueError):
            A._parse_bounded_float("nan", 0.0, 1.0, "fsc_norm")
        with self.assertRaises(ValueError):
            A._parse_bounded_float("1.5", 0.0, 1.0, "fsc_norm")
        with self.assertRaises(ValueError):
            A._valid_jra_rows([{"error": "", "fsc_norm": "nan", "repeat_idx": "0"}])
        with self.assertRaises(ValueError):
            A._valid_jra_rows([{"error": "", "fsc_norm": "0.5", "repeat_idx": "-1"}])

        # All gate variants (gated, gated_format, gated_kernel, full_pipeline) & duplicate/bad-flag rejection
        jr = [
            {"task": "t1", "condition": "faithful_gold_copy", "judge_score": "5"},
            {"task": "t1", "condition": "false_axiom_proofs", "judge_score": "4"},
        ]
        gr = [
            {
                "task": "t1", "condition": "faithful_gold_copy",
                "passes_gate": "1", "passes_gate_and_format": "1",
                "passes_gate_kernel": "1", "passes_all_checks": "1",
            },
            {
                "task": "t1", "condition": "false_axiom_proofs",
                "passes_gate": "1", "passes_gate_and_format": "1",
                "passes_gate_kernel": "0", "passes_all_checks": "0",
            },
        ]
        recs = A._score_e23_dataset(
            "vb",
            jr,
            gr,
            [
                ("judge_only", None),
                ("gated", "passes_gate"),
                ("gated_format", "passes_gate_and_format"),
                ("gated_kernel", "passes_gate_kernel"),
                ("full_pipeline", "passes_all_checks"),
            ],
        )
        m = {(r["condition"], r["pipeline"]): r["mean_score"]["value"] for r in recs}
        self.assertEqual(m[("false_axiom_proofs", "gated")], 4.0)
        self.assertEqual(m[("false_axiom_proofs", "gated_kernel")], 0.0)
        self.assertEqual(m[("false_axiom_proofs", "full_pipeline")], 0.0)

        with self.assertRaises(ValueError):
            A._score_e23_dataset("vb", jr + [jr[0]], gr, [("judge_only", None)])
        jr_no_base = [jr[1]]
        with self.assertRaises(ValueError):
            A._score_e23_dataset("vb", jr_no_base, gr, [("judge_only", None)])
        gr_bad = [dict(gr[0]), {**gr[1], "passes_gate_kernel": "2"}]
        with self.assertRaises(ValueError):
            A._score_e23_dataset("vb", jr, gr_bad, [("gated_kernel", "passes_gate_kernel")])


def _assert_ci_order(tc: unittest.TestCase, st: dict) -> None:
    if st and st.get("value") is not None:
        tc.assertLessEqual(st["ci_lo"], st["value"] + 1e-12)
        tc.assertLessEqual(st["value"], st["ci_hi"] + 1e-12)


class TestGeneratedExp04Artifacts(unittest.TestCase):
    def test_alignment_schemas_and_privacy(self) -> None:
        summary_path = RESULTS_DIR / "summary.json"
        self.assertTrue(summary_path.exists(), "Run scripts/analysis.py before running integration tests.")
        for fname in (
            "equiv_invariance_table.csv",
            "exp_c_ref_swap_table.csv",
            "gaming_monotonicity_table.csv",
            "fig1_vb_fsc_equivalence_and_order_invariance.png",
            "fig2_vb_fsc_gaming_monotonicity.png",
        ):
            p = RESULTS_DIR / fname
            self.assertTrue(p.exists(), f"Missing {fname}")
            if p.suffix == ".png":
                self.assertGreater(p.stat().st_size, 1000)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["alignment_jra"]["passed"])
        self.assertLessEqual(summary["alignment_jra"]["max_abs_diff"], 1e-9)
        self.assertEqual(summary["alignment_jra"]["checks"], 86)

        self.assertTrue(summary["alignment_e23"]["passed"])
        self.assertLessEqual(summary["alignment_e23"]["max_abs_diff"], 0.0051)
        self.assertEqual(summary["alignment_e23"]["checks"], 66)

        with (RESULTS_DIR / "equiv_invariance_table.csv").open("r", encoding="utf-8", newline="") as f:
            eq_rows = list(csv.DictReader(f))
        self.assertEqual(len(eq_rows), 31)
        with (RESULTS_DIR / "exp_c_ref_swap_table.csv").open("r", encoding="utf-8", newline="") as f:
            ec_rows = list(csv.DictReader(f))
        self.assertEqual(len(ec_rows), 5)
        with (RESULTS_DIR / "gaming_monotonicity_table.csv").open("r", encoding="utf-8", newline="") as f:
            gm_rows = list(csv.DictReader(f))
        self.assertEqual(len(gm_rows), 66)

        # Complete CSV parity & CI-bound ordering vs summary.json
        by_a = {r["config_id"]: r for r in summary["part_a_configs"]}
        for row in eq_rows:
            r = by_a[row["config_id"]]
            self.assertEqual(row["effort"], r["effort"])
            self.assertEqual(int(row["n_t1_equiv_same_effort"]), r["V_equiv_same_effort"]["n"])
            self.assertEqual(int(row["n_mismatched_effort_pairs"]), r["n_mismatched_equiv_effort"])
            for csv_k, rec_k in [
                ("cm", "cm"), ("cd", "cd"), ("A_G", "A_G"), ("V_equiv", "V_equiv"),
                ("V_equiv_same_effort", "V_equiv_same_effort"),
                ("V_rename_exact", "V_rename_exact"), ("V_order", "V_order"),
                ("fc", "fc"), ("V_rep_ab", "V_rep_ab"),
            ]:
                st = r[rec_k]
                _assert_ci_order(self, st)
                if st["value"] is None:
                    self.assertEqual(row[csv_k], "")
                else:
                    self.assertAlmostEqual(float(row[csv_k]), st["value"], places=12)

        by_c = {r["config_id"]: r for r in summary["part_a_exp_c"]}
        for row in ec_rows:
            r = by_c[row["config_id"]]
            self.assertEqual(int(row["n_pairs"]), r["n_pairs"])
            for field in ("V_ref_swap", "mean_abs_fsc_shift", "signed_shift_RA_minus_RB", "harsh_flip_nodt_eq1"):
                st = r[field]
                _assert_ci_order(self, st)
                if st["value"] is None:
                    self.assertEqual(row[field], "")
                else:
                    self.assertAlmostEqual(float(row[field]), st["value"], places=12)

        by_b = {(r["dataset"], r["condition"], r["pipeline"]): r for r in summary["part_b_records"]}
        for row in gm_rows:
            r = by_b[(row["dataset"], row["condition"], row["pipeline"])]
            self.assertEqual(int(row["gate_pass_count"]), r["gate_pass_count"])
            for field in ("mean_score", "mean_drop", "V_mono", "high_credit_ge3"):
                st = r[field]
                _assert_ci_order(self, st)
                if st["value"] is None:
                    self.assertEqual(row[field], "")
                else:
                    self.assertAlmostEqual(float(row[field]), st["value"], places=12)
                    self.assertAlmostEqual(float(row[f"{field}_ci_lo"]), st["ci_lo"], places=12)
                    self.assertAlmostEqual(float(row[f"{field}_ci_hi"]), st["ci_hi"], places=12)

        if A.DEFAULT_JRA_DIR.exists() and A.DEFAULT_E23_DIR.exists():
            jra_fresh = A.analyze_jra(A.DEFAULT_JRA_DIR)
            e23_fresh = A.analyze_e23(A.DEFAULT_E23_DIR)
            self.assertEqual(jra_fresh["alignment_jra"], summary["alignment_jra"])
            self.assertEqual(jra_fresh["equiv_triggers"], summary["equiv_triggers"])
            self.assertEqual(jra_fresh["part_a_configs"], summary["part_a_configs"])
            self.assertEqual(jra_fresh["part_a_exp_c"], summary["part_a_exp_c"])
            self.assertEqual(e23_fresh["alignment_e23"], summary["alignment_e23"])
            self.assertEqual(e23_fresh["game_triggers"], summary["game_triggers"])
            self.assertEqual(e23_fresh["records"], summary["part_b_records"])

        # Privacy & secret guard across generated results/
        email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        for p in list(RESULTS_DIR.glob("*")):
            if p.suffix in (".json", ".csv", ".md"):
                txt = p.read_text(encoding="utf-8")
                self.assertNotIn(".lean", txt)
                self.assertNotIn("rationale", txt)
                self.assertNotIn("/Users/", txt)
                self.assertNotIn("/lfs/", txt)
                self.assertIsNone(email_re.search(txt), f"Unexpected email in {p}")


if __name__ == "__main__":
    unittest.main()
