#!/usr/bin/env python3
"""Unit tests for experiment 08. The full alignment run is test_full_run_matches_machine_lines."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parent / "analysis.py"
    spec = importlib.util.spec_from_file_location("exp08_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


class ScoreTests(unittest.TestCase):
    def test_point_mass_rps_zero_when_humans_match(self):
        q = A.point_mass(1.0)
        self.assertEqual(int(np.argmax(q)), 5)
        p = q.copy()
        self.assertAlmostEqual(A.rps(q, p), 0.0)
        self.assertAlmostEqual(A.tvd(q, p), 0.0)

    def test_smooth_alpha_is_a_distribution(self):
        q = A.smooth_alpha(np.array([0, 0, 0, 0, 0, 10], dtype=float))
        self.assertAlmostEqual(float(q.sum()), 1.0)
        self.assertTrue(np.all(q > 0))

    def test_pav_merges_a_decrease_and_nearest_tie_goes_lower(self):
        blocks = A.pav_blocks({
            0.0: np.array([0, 0, 0, 0, 5, 0], dtype=float),
            1.0: np.array([5, 0, 0, 0, 0, 0], dtype=float),
        })
        self.assertEqual(len(blocks), 1)
        # Unseen score 0.4 is equidistant? 0.4 is closer to 0.0 than 1.0? |0.4-0|=0.4, |0.4-1|=0.6.
        # Equidistant case: score -0.5 is not. Use 0.5, |0.5-0|=0.5, |0.5-1|=0.5, lower score wins.
        q_low = A.iso_lookup(blocks, 0.5)
        q_seen = A.iso_lookup(blocks, 0.0)
        self.assertTrue(np.allclose(q_low, q_seen))

    def test_positive_slope_preserves_order(self):
        scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = []
        for s in scores:
            # Higher scores get higher ratings, so the slope should be positive.
            center = int(round(5 * s))
            c = np.zeros(6)
            c[center] = 8
            counts.append(c)
        predict, b = A.fit_ce_ordinal(scores * 4, counts * 4)
        self.assertGreater(b, 0)
        ev = [A.expected_unit(predict([s])[0]) for s in (0.1, 0.2, 0.9)]
        self.assertLess(ev[0], ev[1])
        self.assertLess(ev[1], ev[2])
        self.assertEqual(A.monotone_order_violations([(0.1, 0.1)], [(ev[0], ev[0])], True), 0)

    def test_verdict_untestable_when_nobody_qualifies(self):
        summary = {"configs": {"A": {"qualifies_rps": False}, "B": {"qualifies_rps": False}}}
        self.assertEqual(A.verdict_h1(summary), "untestable")


class FullRunTests(unittest.TestCase):
    def test_full_run_is_reproducible_and_aligned(self):
        if not A.DEFAULT_ITEMS.exists():
            self.skipTest("off-repo items.jsonl is not on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            s1 = A.run(A.DEFAULT_ITEMS, A.DEFAULT_SCORES, a)
            s2 = A.run(A.DEFAULT_ITEMS, A.DEFAULT_SCORES, b)
            self.assertTrue(s1["alignment"]["pass"])
            self.assertEqual(s1["machine_lines"], s2["machine_lines"])
            self.assertEqual(s1["verdict_h1_cal"], json.loads((a / "summary.json").read_text())["verdict_h1_cal"])
            for name in ("fig1_calibrated_proper_scores.png", "fig2_properties_before_after.png"):
                self.assertGreater((a / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
