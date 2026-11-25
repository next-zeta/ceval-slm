#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归测试：评测题数与每题生成次数必须独立。无需模型或网络。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import build_summary
from report import render


class SummaryTests(unittest.TestCase):
    def test_question_count_and_generation_count_are_separate(self):
        for generations in (1, 3):
            with self.subTest(generations=generations):
                rows = [
                    {
                        "subject": "accountant",
                        "correct": i < 2,
                        "format_valid": True,
                        "truncated": False,
                        "elapsed_s": 0.5,
                        "n_calls": generations,
                        "output_tokens": 4 * generations,
                    }
                    for i in range(5)
                ]
                summary = build_summary(rows, {
                    "template": "base",
                    "generations_per_question": generations,
                })
                self.assertEqual(summary["n_samples"], 5)
                self.assertEqual(summary["correct"], 2)
                self.assertEqual(summary["accuracy"], 2 / 5)
                self.assertEqual(summary["generations_per_question"], generations)
                self.assertEqual(summary["total_n_calls"], 5 * generations)
                self.assertIn("base/0/{}/1".format(generations), render([summary]))


if __name__ == "__main__":
    unittest.main()
