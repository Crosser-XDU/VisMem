# VisMem Reproduction Notes

The public code in this repository was not directly runnable as-is and does not fully match the algorithm described in the VisMem paper. The fixes here keep the repository's existing training structure, while making the runnable behavior explicit.

## Fixed Engineering Issues

- Config parsing now reads the `vismem:` key instead of a nonexistent `main:` key.
- Training scripts use the actual package entrypoints, `python -m main.cli.*`.
- Qwen2.5-VL processor inputs are built with a chat template when the prompt does not already contain image tokens.
- Added special tokens are synchronized back into `processor.tokenizer`.
- Stage I no longer freezes the LoRA memory-former parameters by accident.
- Stage I baseline loss keeps the multimodal image tensors instead of evaluating a text-only concatenation.
- Target padding is masked in Stage I and Stage II losses.
- Stage II no longer has the `reward_ema` local-variable crash.
- Stage II now updates inside the sample loop and uses candidate groups so the advantage is not always zero.
- Stage II trains untied output embeddings for the invocation/end tokens when the base model has a separate LM head.
- Stage II sets `stage2_weight_decay: 0.0` by default so AdamW does not bypass gradient hooks and decay the entire embedding matrix.
- Checkpoint saving is main-rank only under distributed training.

## Paper-Aligned Mode

`configs/vismem_qwen25vl7b_paper.yaml` enables `training.paper_aligned: true`. In that mode:

- Stage I samples candidate trajectories with forced short/long memory invocations, computes reward improvement over a no-memory baseline trajectory, and applies a clipped group-relative policy objective to the memory formation path.
- Stage II samples candidate trajectories from the learned invocation policy, applies the reverse-memory-type and below-group-mean penalties, and replays latent memory insertions when computing policy log probabilities.
- The paper appendix parameters are exposed in YAML: `group_size`, `clip_ratio`, `stage1_kl_beta`, `kl_beta`, and `penalty_alpha`.
- Rollouts cache old log probabilities before updates. `paper_policy_epochs` controls how many optimizer updates reuse the same rollout, so the clipped ratio is meaningful after the first update.

## Remaining Algorithmic Gaps

The compatibility config still keeps the public repository's simpler Stage I memory-formation loss, which is a teacher-forced answer loss with a detached baseline term. Use the paper config for the GRPO-style path.

The paper also says the short-memory LoRA is attached to the vision encoder and projected through the original vision projector, while the public code uses LoRA adapters on the base language model path for both short and long memory formation. This is a substantive architecture mismatch and should be treated as future reproduction work, not a small bug fix.

The exact delimiter curriculum from Stage I is not fully specified in executable detail. The paper-aligned implementation exposes `stage1_force_position` and defaults to random forced invocation positions.

The KL term is implemented as a sampled-trajectory approximation. The paper specifies a KL penalty, but does not provide code for the exact token-level estimator used with latent memory insertion.

Reverse-memory penalties reuse the sampled trajectory's invocation decode steps and swap memory types before counterfactual generation. This keeps the memory decision schedule fixed, though the continuation still has to be generated under the counterfactual memory state.

## Source Anchors

- Paper: https://arxiv.org/abs/2511.11007
- Official public repository: https://github.com/Crosser-XDU/VisMem
