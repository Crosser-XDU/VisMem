from __future__ import annotations

import unittest

try:
    import torch
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PaperGRPOTests(unittest.TestCase):
    def test_sampled_kl_keeps_current_policy_gradient(self):
        from main.trainer.paper_grpo import sampled_kl_from_logps

        logp = torch.tensor([0.2], requires_grad=True)
        ref_logp = torch.tensor([0.0])
        loss = sampled_kl_from_logps(logp, ref_logp)
        loss.backward()
        self.assertIsNotNone(logp.grad)
        self.assertGreater(abs(float(logp.grad.item())), 0.0)

    def test_clipped_grpo_uses_fixed_old_logp(self):
        from main.trainer.paper_grpo import clipped_grpo_loss

        logp = torch.tensor([0.3], requires_grad=True)
        old_logp = torch.tensor([0.0])
        adv = torch.tensor([1.0])
        loss = clipped_grpo_loss(logp, old_logp, adv, clip_ratio=0.2)
        loss.backward()
        self.assertIsNotNone(logp.grad)
        self.assertEqual(float(logp.grad.item()), 0.0)

    def test_invocation_schedule_uses_decode_steps_not_auto_end_tokens(self):
        from main.model.model import VisMemModel

        class Dummy:
            short_invoke_id = 1
            short_end_id = 2
            long_invoke_id = 3
            long_end_id = 4

            def _memory_token_ids(self, mem_type):
                if mem_type == "short":
                    return self.short_invoke_id, self.short_end_id
                return self.long_invoke_id, self.long_end_id

        ids = torch.tensor([[1, 2, 99, 3, 4]])
        schedule = VisMemModel.extract_invocation_schedule(Dummy(), ids, reverse_types=True)
        self.assertEqual(schedule, [(0, "long"), (2, "short")])


if __name__ == "__main__":
    unittest.main()
