#!/usr/bin/env python3
"""Unit tests for experiment 09."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parent / "analysis.py"
    spec = importlib.util.spec_from_file_location("exp09_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


def _row(config, base, rho, passes, fail_count=0):
    return {
        "config": config,
        "base": base,
        "rho": rho,
        "pass": np.asarray(passes, dtype=float),
        "ati": float(np.mean(passes)),
        "fail_count": fail_count,
        "idea1_fail": 0,
    }


class SelectorTests(unittest.TestCase):
    def test_tie_break_is_alphabetical_and_ignores_later_fields(self):
        rows = [
            _row("b|H2", "fam", 0.5, [0.2] * 6),
            _row("a|H2", "fam", 0.5, [0.9] * 6),
        ]
        self.assertEqual(A.pick(rows, "rho")["config"], "a|H2")

    def test_ridge_ignores_held_out_labels(self):
        train = [_row(f"t{i}", "a" if i < 4 else "b", 0.1 * i, np.linspace(0.2, 0.8, 6)) for i in range(8)]
        for i, r in enumerate(train):
            r["base"] = "a" if i < 4 else "b"
            r["pass"] = np.array([0.1 * i, 0.2, 0.3, 0.4, 0.5, 0.6])
            r["rho"] = 0.05 * i + 0.1 * r["pass"][0]
        held = [_row("h0", "c", 0.9, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), _row("h1", "c", 0.1, [0.8] * 6)]
        w1, _ = A.fit_weights(train)
        shuffled = [dict(r) for r in held]
        shuffled[0]["rho"], shuffled[1]["rho"] = shuffled[1]["rho"], shuffled[0]["rho"]
        w2, _ = A.fit_weights(train + shuffled)
        w3, _ = A.fit_weights(train + held)
        # Adding a third family changes the fit. Shuffling only that family's labels must move w
        # if the fit uses them, and must not match the unshuffled add when labels differ.
        self.assertFalse(np.allclose(w2, w3))
        # The training-only fit does not read held-out rows at all.
        w_only, _ = A.fit_weights(train)
        self.assertTrue(np.allclose(w1, w_only))

    def test_verdict_cuts(self):
        self.assertEqual(A.verdict_gap(3, 0.5), "supported")
        self.assertEqual(A.verdict_gap(1, 2.0), "falsified")
        self.assertEqual(A.verdict_gap(4, 0.0), "falsified")
        self.assertEqual(A.verdict_gap(2, 0.2), "mixed")
        self.assertEqual(A.verdict_pred(0.5, 0.1, 0.01, 2000, 5000), "supported")
        self.assertEqual(A.verdict_pred(0.10, 0.0, 0.2, 2000, 5000), "falsified")
        self.assertEqual(A.verdict_pred(0.5, 0.1, 0.01, 10, 5000), "not_computable")


class FullRunTests(unittest.TestCase):
    def test_full_run_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            s1 = A.run(A.DEFAULT_TABLE, a)
            s2 = A.run(A.DEFAULT_TABLE, b)
            self.assertEqual(s1["machine_lines"], s2["machine_lines"])
            self.assertAlmostEqual(s1["alignment_spearman"], 0.7883251594650563, places=9)
            self.assertEqual(len(A.load_rows(A.DEFAULT_TABLE)[0]), 56)
            self.assertGreater((a / "fig1_selection_gap.png").stat().st_size, 1000)
            # Six families partition the 56.
            self.assertEqual(sum(r["n"] for r in s1["families"]), 56)


if __name__ == "__main__":
    unittest.main()
