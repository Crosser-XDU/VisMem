from __future__ import annotations

import unittest

try:
    import torch
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class QwenVLInputTests(unittest.TestCase):
    def test_extends_mm_token_types_for_appended_text(self):
        from main.utils.qwen_vl import extend_qwen_vl_inputs

        pixel_values = torch.randn(4, 8)
        inputs = {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]], dtype=torch.int32),
            "pixel_values": pixel_values,
        }
        full_ids = torch.tensor([[10, 11, 12, 13, 14]])
        full_mask = torch.ones(1, 5, dtype=torch.long)

        extended = extend_qwen_vl_inputs(inputs, full_ids, full_mask)

        self.assertEqual(extended["mm_token_type_ids"].tolist(), [[0, 1, 0, 0, 0]])
        self.assertIs(extended["pixel_values"], pixel_values)
        self.assertIs(extended["input_ids"], full_ids)
        self.assertIs(extended["attention_mask"], full_mask)


if __name__ == "__main__":
    unittest.main()
