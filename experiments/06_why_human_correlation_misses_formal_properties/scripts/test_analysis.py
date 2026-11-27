#!/usr/bin/env python3
"""Checks for experiment 06. Synthetic cases, alignment to experiments 03 and 05, and the unrun API script."""
from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
EXP = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import analysis as A  # noqa: E402
import judge_implementation_rubric as J  # noqa: E402

FIXTURE = SCRIPTS / "fixtures" / "probes_min.jsonl"
EXP05 = A.EXP05


def indep_class(id1: float, bug: float) -> str:
    delta = bug - id1
    if delta > 0.05 + 1e-9:
        return "inversion"
    if delta < -(0.05 + 1e-9):
        return "penalized"
    if id1 >= 0.95 - 1e-9:
        return "ceiling_tie"
    if id1 <= 0.20 + 1e-9 and bug <= 0.20 + 1e-9:
        return "floor_tie"
    return "mid_tie"


class TestPairClasses(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(A.classify_pair(1.0, 1.0), "ceiling_tie")
        self.assertEqual(A.classify_pair(1.0, 0.95), "ceiling_tie")
        self.assertEqual(A.classify_pair(1.0, 0.949), "penalized")
        self.assertEqual(A.classify_pair(1.0, 0.7), "penalized")
        self.assertTrue(A.is_material(1.0, 0.7))
        self.assertFalse(A.is_material(1.0, 0.8))
        self.assertEqual(A.classify_pair(0.5, 0.56), "inversion")
        self.assertEqual(A.classify_pair(0.5, 0.55), "mid_tie")
        self.assertEqual(A.classify_pair(0.5, 0.5), "mid_tie")
        self.assertEqual(A.classify_pair(0.2, 0.2), "floor_tie")
        self.assertEqual(A.classify_pair(0.2, 0.0), "penalized")
        self.assertEqual(A.classify_pair(0.96, 1.0), "ceiling_tie")
        with self.assertRaises(ValueError):
            A.classify_pair(1.2, 0.0)

    def test_partition_on_a_grid(self):
        for i in range(11):
            for j in range(11):
                got = A.classify_pair(i / 10, j / 10)
                self.assertIn(got, A.CLASSES)
                self.assertEqual(got, indep_class(i / 10, j / 10))


class TestVerdictRules(unittest.TestCase):
    def test_overall(self):
        self.assertEqual(A.overall_verdict("supported", "supported", "mixed"), "supported")
        self.assertEqual(A.overall_verdict("falsified", "falsified", "mixed"), "falsified")
        self.assertEqual(A.overall_verdict("supported", "falsified", "mixed"), "mixed")
        self.assertEqual(A.overall_verdict("supported", "supported", "not_computable"), "not_computable")

    def test_material_and_range_cuts(self):
        supported = {"value": 0.10, "ci_lo": 0.0, "ci_hi": 0.20, "valid_draws": A.DRAWS}
        falsified = {"value": 0.40, "ci_lo": 0.20, "ci_hi": 0.60, "valid_draws": A.DRAWS}
        mixed = {"value": 0.22, "ci_lo": 0.10, "ci_hi": 0.35, "valid_draws": A.DRAWS}
        self.assertEqual(A.verdict_material(supported), "supported")
        self.assertEqual(A.verdict_material(falsified), "falsified")
        self.assertEqual(A.verdict_material(mixed), "mixed")
        self.assertEqual(A.verdict_range(0.10), "supported")
        self.assertEqual(A.verdict_range(0.02), "falsified")
        self.assertEqual(A.verdict_range(0.05), "mixed")

    def test_arm_i_rules(self):
        block = {"value": 0.10, "ci_lo": 0.02, "ci_hi": 0.30}
        self.assertEqual(
            A.verdict_arm_i(block, 0.40, 0.95, 37, 24, "claude-opus-5-5", "stanford-claude"),
            "supported",
        )
        fail = {"value": 0.55, "ci_lo": 0.30, "ci_hi": 0.75}
        self.assertEqual(
            A.verdict_arm_i(fail, 0.05, 0.95, 37, 24, "claude-opus-5-5", "vals-claude-opus"),
            "falsified",
        )
        self.assertEqual(A.verdict_arm_i(block, 0.40, 0.95, 3, 24, "claude-opus-5-5", "stanford-claude"), "incomplete")
        self.assertEqual(A.verdict_arm_i(block, 0.40, 0.20, 37, 24, "claude-opus-5-5", "stanford-claude"), "not_valid")

    def test_split_absent_channel(self):
        weak = {"value": -0.70, "ci_lo": -0.80, "ci_hi": -0.50, "p_val": 0.001, "valid_draws": 2000, "valid_perm": 5000}
        bug = {"value": -0.05, "ci_lo": -0.20, "ci_hi": 0.10, "p_val": 0.40, "valid_draws": 2000, "valid_perm": 5000}
        self.assertEqual(A.verdict_split(weak, bug), "supported")
        bug_track = {"value": -0.60, "ci_lo": -0.70, "ci_hi": -0.40, "p_val": 0.001, "valid_draws": 2000, "valid_perm": 5000}
        weak_flat = {"value": -0.05, "ci_lo": -0.20, "ci_hi": 0.10, "p_val": 0.40, "valid_draws": 2000, "valid_perm": 5000}
        self.assertEqual(A.verdict_split(weak_flat, bug_track), "falsified")


class TestStatistics(unittest.TestCase):
    def test_slack_example(self):
        got = A.slack_example()
        self.assertEqual(got["spearman_of_medians"], 1.0)
        self.assertEqual(got["v_rep"], 1.0)

    def test_cluster_interval_is_deterministic(self):
        pairs = [("a", 0.0), ("a", 1.0), ("b", 0.2), ("c", 0.8), ("c", 0.8), ("d", 0.1)]
        first = A.cluster_stat(pairs, A._mean, seed=20261124, draws=200)
        second = A.cluster_stat(pairs, A._mean, seed=20261124, draws=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first["ci_lo"], first["value"])
        self.assertLessEqual(first["value"], first["ci_hi"])
        self.assertEqual(first["n_tasks"], 4)
        self.assertEqual(first["n"], 6)

    def test_partial_spearman_known_cases(self):
        rho = [float(i) for i in range(40)]
        rng = np.random.default_rng(0)
        weak = (np.asarray(rho) + rng.normal(0.0, 5.0, size=40)).tolist()
        bug = rng.normal(size=40).tolist()
        self.assertIsNone(A.partial_spearman(rho, rho, rho))
        both = A.partial_bundle(rho, weak, bug, draws=200, perm_draws=300)
        noise = A.partial_bundle(rho, bug, weak, draws=200, perm_draws=300)
        self.assertGreater(both["value"], 0.90)
        self.assertGreater(both["ci_lo"], 0.8)
        self.assertLess(both["p_val"], 0.05)
        self.assertIsNotNone(noise["value"])
        self.assertLess(abs(noise["value"]), 0.45)

    def test_constants_match_the_prereg_cuts(self):
        self.assertEqual(A.MATERIAL_SUPPORT_POINT_MAX, 0.15)
        self.assertEqual(A.MATERIAL_SUPPORT_CI_HI_MAX, 0.30)
        self.assertEqual(A.MATERIAL_FALSIFY_POINT_MIN, 0.30)
        self.assertEqual(A.RANGE_SUPPORT_MIN, 0.10)
        self.assertEqual(A.RANGE_FALSIFY_MAX, 0.02)
        self.assertEqual(A.SPLIT_TRACK_POINT, -0.40)
        self.assertEqual(A.SPLIT_TRACK_CI_HI, -0.20)
        self.assertEqual(A.SEED, 20261124)
        self.assertEqual(A.DRAWS, 2000)
        text = (EXP / "PREREG.md").read_text(encoding="utf-8")
        self.assertIn("point `≤ 0.15`", text)
        self.assertIn("median `≥ 0.10`", text)
        self.assertIn("point `≤ −0.40`", text)


class TestApiScript(unittest.TestCase):
    def test_rubric_matches_prereg(self):
        text = (EXP / "PREREG.md").read_text(encoding="utf-8")
        match = re.search(
            r"<!-- RUBRIC_I_BEGIN -->\n```text\n(.*?)\n```\n<!-- RUBRIC_I_END -->",
            text,
            re.S,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), J.RUBRIC_I)
        self.assertIn("score must be at most 0.4", J.RUBRIC_I)

    def test_dry_run_does_not_call_the_router(self):
        buf = io.StringIO()
        argv = ["judge_implementation_rubric.py", "--dry-run", "--probes", str(FIXTURE)]
        with patch.object(J.subprocess, "run", side_effect=AssertionError("router called")):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(buf):
                    J.main()
        report = json.loads(buf.getvalue())
        self.assertEqual(report["n_in"], 16)
        self.assertEqual(report["n_kept"], 15)
        self.assertNotIn("CRIT", report["kinds_kept"])
        self.assertGreaterEqual(report["n_batches"], 2)
        self.assertTrue(all(len(tasks) == len(set(tasks)) for tasks in report["batch_tasks"]))
        self.assertTrue(all(size <= 6 for size in report["batch_sizes"]))
        self.assertTrue(report["prompt_contains_rubric"])
        self.assertTrue(report["prompt_contains_item_header"])
        self.assertFalse(report["router_called"])

    def test_refuses_banned_lane_and_repo_output(self):
        with self.assertRaises(SystemExit):
            J.validate_lanes(["stanford-claude", "agy-test"])
        with self.assertRaises(SystemExit):
            J.validate_lanes(["personal-llama-4"])
        with self.assertRaises(SystemExit):
            J.assert_out_outside_repo(EXP)
        buf = io.StringIO()
        argv = [
            "judge_implementation_rubric.py",
            "--probes",
            str(FIXTURE),
            "--lanes",
            "agy-hidden",
            "--out",
            "/tmp/exp06-should-not-matter",
            "--confirm-plan-spend",
        ]
        with patch.object(J.subprocess, "run", side_effect=AssertionError("router called")):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(buf):
                    with self.assertRaises(SystemExit):
                        J.main()

    def test_parse_text_parse_scores_and_completed_batches(self):
        ndjson = "\n".join([
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": '[{"item": 1, "score": 0.2}]'}]}}),
            json.dumps({"type": "result", "result": 'notes [not json] then [{"item": 1, "score": 1.0}, {"item": 2, "score": 0.4}]'}),
        ])
        texts = J.parse_text(ndjson)
        self.assertEqual(len(texts), 2)
        parsed = J.parse_scores(texts, n_items=2)
        self.assertEqual(parsed, {1: 1.0, 2: 0.4})
        two_arrays = 'first [{"item": 9, "score": 0.1}] middle text second [{"item": 1, "score": 0.8}, {"item": 2, "score": 0.0}]'
        self.assertEqual(J.parse_scores([two_arrays], n_items=2), {1: 0.8, 2: 0.0})
        with tempfile.TemporaryDirectory() as tmp:
            resf = Path(tmp) / "scores_I.jsonl"
            resf.write_text(
                "\n".join([
                    json.dumps({"batch": 0, "probe_ids": ["p1", "p2"], "scores": {"p1": 1.0}}),
                    json.dumps({"batch": 1, "probe_ids": ["p3", "p4"], "scores": {"p3": 1.0, "p4": 0.4}}),
                ]) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(J.completed_batches(resf), {1})


class TestLoggedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = A.build_report()
        cls.repeat = A.build_report()

    def test_deterministic(self):
        self.assertEqual(A.canonical_lines(self.report), A.canonical_lines(self.repeat))

    def test_alignment_and_partition(self):
        self.assertEqual(self.report["verdicts"]["overall"], A.overall_verdict(
            self.report["verdicts"]["h_material"],
            self.report["verdicts"]["h_range"],
            self.report["verdicts"]["h_split"],
        ))
        self.assertEqual(self.report["arm_i"]["status"], "not_run")
        self.assertEqual(self.report["new_model_calls"], 0)
        self.assertLess(self.report["exp03"]["spearman_abs_diff"], 1e-6)
        self.assertEqual(self.report["exp03"]["n_high"], 22)
        for key in A.CONFIG_ORDER:
            block = self.report["configs"][key]
            self.assertAlmostEqual(sum(block["classes"][name]["value"] for name in A.CLASSES), 1.0)
            self.assertEqual(sum(block["n_class"].values()), block["n_bug_pairs"])
            self.assertTrue(all(v == 0 for v in block["disjoint_crit_intersection"].values()))
            self.assertGreater(block["n_tasks_in_both_crit_and_bug"], 0)
            rows = A.load_score_csv(EXP05 / f"scores_{key}.csv")
            pairs = A.bug_pair_rows(rows, A.index_id1(rows))
            share = sum(1 for pair in pairs if pair["drop"] > 0.20 + 1e-9) / len(pairs)
            self.assertAlmostEqual(share, block["material_drop_gt_0_20"]["value"])
            for pair in pairs:
                self.assertEqual(pair["pair_class"], indep_class(pair["id1"], pair["bug"]))
            if block["median_drop_corollary_zero"]:
                self.assertLessEqual(abs(block["median_drop"]["value"]), 1e-9)

    def test_partial_matches_an_independent_implementation(self):
        data = json.loads((A.EXP03 / "summary.json").read_text(encoding="utf-8"))
        units = [u for u in data["units"] if u["kind"] == "llm" and u["rho_heldout"] is not None]
        rho = [float(u["rho_heldout"]) for u in units]
        weak = [float(u["V_weak"]["value"]) for u in units]
        bug = [float(u["V_bug"]["value"]) for u in units]
        rx = A.rankdata(rho, method="average")
        ry = A.rankdata(weak, method="average")
        rz = A.rankdata(bug, method="average")

        def resid(target, covariate):
            design = np.column_stack([np.ones(len(target)), covariate])
            coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            return target - design @ coef

        got = float(A.pearsonr(resid(rx, rz), resid(ry, rz)).statistic)
        stored = self.report["exp03"]["partial_rho_vweak_given_vbug"]["value"]
        self.assertAlmostEqual(got, stored, places=12)

    def test_outputs_and_privacy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            lines = A.write_outputs(self.report, Path(tmp))
            self.assertEqual(lines[0].split()[0], "OVERALL")
            for name in ("summary.json", "bug_pairs.csv", "fig1_bug_pair_classes.png", "fig2_repeat_range_vs_human_correlation.png"):
                path = Path(tmp) / name
                self.assertTrue(path.exists(), name)
            self.assertEqual((Path(tmp) / "fig1_bug_pair_classes.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            summary = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            json.dumps(summary, allow_nan=False)
            blob = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in Path(tmp).glob("*") if path.suffix != ".png")
            for banned in ("api_key", "sk-", "BEGIN PRIVATE", "BEGIN OPENSSH", "password"):
                self.assertNotIn(banned, blob)
            with (Path(tmp) / "bug_pairs.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 37 * 3)

    def test_results_quote_canonical_lines_when_present(self):
        results = EXP / "RESULTS.md"
        lines_path = EXP / "results" / "canonical_lines.txt"
        if not results.exists() or not lines_path.exists():
            self.skipTest("RESULTS.md is written after the CPU run")
        body = results.read_text(encoding="utf-8")
        missing = [line for line in lines_path.read_text(encoding="utf-8").splitlines() if line and line not in body]
        self.assertEqual(missing, [])
        self.assertNotIn("Not run yet", body)

    def test_arm_i_parity_when_scored(self):
        scores_i = EXP / "results" / "scores_I.csv"
        summary_path = EXP / "results" / "summary.json"
        if not scores_i.exists() or not summary_path.exists():
            self.skipTest("scores_I.csv not present")
        arm_i = A.summarize_arm_i_csv(scores_i, "claude-opus-5-5", "vals-claude-opus")
        self.assertEqual(arm_i["status"], "scored")
        self.assertEqual(arm_i["n_id1"], 24)
        self.assertEqual(arm_i["n_bug"], 37)
        self.assertEqual(arm_i["verdict"], "supported")
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["arm_i"]["verdict"], arm_i["verdict"])
        self.assertAlmostEqual(stored["arm_i"]["v_bug"]["value"], arm_i["v_bug"]["value"], places=12)
        self.assertAlmostEqual(stored["arm_i"]["mean_drop"], arm_i["mean_drop"], places=12)
        self.assertAlmostEqual(stored["arm_i"]["id1_mean"], arm_i["id1_mean"], places=12)
        report = A.build_report()
        report["arm_i"] = arm_i
        self.assertEqual(
            A.canonical_lines(report),
            (EXP / "results" / "canonical_lines.txt").read_text(encoding="utf-8").splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
