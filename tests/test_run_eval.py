#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用模拟设备检查计时边界与 EOS 处理，无需权重或 GPU。"""
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_eval import DEFAULT_CONFIG, generate_samples


class GenerationTests(unittest.TestCase):
    def test_gpu_sync_surrounds_generation_and_cpu_skips_sync(self):
        for backend in ("cpu", "cuda", "mps"):
            with self.subTest(backend=backend):
                device = SimpleNamespace(type=backend)
                events = []

                input_ids = Mock(shape=(2, 4))
                input_ids.to.return_value = input_ids
                mask = Mock()
                mask.to.return_value = mask
                mask.sum.return_value.tolist.return_value = [3, 4]
                tokenizer = Mock(return_value={"input_ids": input_ids, "attention_mask": mask}, pad_token_id=11)

                def decode(ids, skip_special_tokens):
                    events.append("decode")
                    return " A "

                tokenizer.decode.side_effect = decode
                output = MagicMock(shape=(2, 7))
                output.__getitem__.side_effect = [Mock(tolist=lambda: [5, 6, 10]), Mock(tolist=lambda: [5, 6, 7])]

                def generate(**kwargs):
                    events.append("generate")
                    return output

                model = Mock(generation_config=SimpleNamespace(eos_token_id=[9, 10]))
                model.generate.side_effect = generate
                torch = Mock(inference_mode=nullcontext)
                torch.cuda.synchronize.side_effect = lambda _: events.append("sync")
                torch.mps.synchronize.side_effect = lambda: events.append("sync")

                def clock():
                    events.append("clock")
                    return 1.0 if events.count("clock") == 1 else 3.0

                with patch("run_eval.time.perf_counter", side_effect=clock):
                    rows, seconds = generate_samples(model, tokenizer, ["a", "b"],
                                                     dict(DEFAULT_CONFIG, max_new_tokens=3), device, torch)
                if backend == "cpu":
                    self.assertEqual(events, ["clock", "generate", "clock", "decode", "decode"])
                else:
                    self.assertEqual(events, ["sync", "clock", "generate", "sync", "clock", "decode", "decode"])
                self.assertEqual(torch.cuda.synchronize.call_count, 2 if backend == "cuda" else 0)
                self.assertEqual(torch.mps.synchronize.call_count, 2 if backend == "mps" else 0)
                self.assertEqual(seconds, 2.0)
                self.assertEqual(rows[0]["generated_token_ids"], [5, 6, 10])
                self.assertEqual(rows[0]["text_for_parsing"], "A")
                self.assertEqual(rows[0]["output_tokens"], 2)
                self.assertFalse(rows[0]["truncated"])
                self.assertTrue(rows[1]["truncated"])


if __name__ == "__main__":
    unittest.main()
