<!-- .github/pull_request_template.md -->

## 📌 Description

### Motivation

vLLM's expert-parallel routing replay feature (`RoutedExpertsCapturer`) needs to record which experts each token is routed to during MoE inference. This works fine for the non-fused path where `BaseRouter.select_experts()` is called and `capture_fn` fires. However, the **TRTLLM-GEN fused MoE path is monolithic** — `Fp8MoEMethod.apply_monolithic()` calls FlashInfer's `trtllm_fp8_block_scale_moe` directly, bypassing the router entirely. This means `capture_fn` is never invoked and the `RoutedExpertsCapturer` never receives expert IDs from this code path.

Without this change, any vLLM deployment using the TRTLLM-GEN fused MoE backend cannot use routing replay, which is required for expert-parallel inference and load-balancing analytics.

### What this PR does

Adds an optional `routing_replay_out` parameter to FlashInfer's fused routing and MoE kernels. When a pre-allocated `int16` tensor of shape `[num_tokens, topk]` is provided, the CUDA routing kernel writes selected expert IDs per token directly into it during routing — inside the same fused kernel call that computes the MoE output. When `None` (the default), the kernel skips the write entirely with zero overhead.

**API surface (all backward-compatible, default `None`):**
- `flashinfer.fused_moe.fused_topk_deepseek(..., routing_replay_out=None)`
- `flashinfer.fused_moe.trtllm_fp8_block_scale_moe(..., routing_replay_out=None)`

**Changes across the stack:**
- **CUDA kernels**: plumb `routing_replay_out` through `deepseek_v3_topk_kernel`, `routingMainKernel`, `invokeNoAuxTc`, and all three launcher classes (`FusedMoeLauncher`, `Fp8BlockScaleLauncher`, `FP4BlockScaleLauncher`)
- **C++ bindings**: add `int16_code` constant, `Optional<TensorView>` parameter with shape/dtype validation in `NoAuxTc` and `trtllm_fp8_block_scale_moe`
- **Python API**: add `routing_replay_out` to `fused_topk_deepseek`, `trtllm_fp8_block_scale_moe`, and their `torch.compile` registrations (`custom_op`, `fake_op`, `mutates_args`)
- **Input validation**: dtype and shape checks in `_check_dsv3_fused_routing_supported`

### vLLM integration status

This feature has been tested end-to-end with vLLM's `RoutedExpertsCapturer` and the corresponding vLLM integration PR is in progress toward merging into vLLM main. The vLLM-side change is minimal (3 files, ~15 lines) — it slices the existing `_RoutedExpertsDeviceCache` buffer and passes it as `routing_replay_out`.

## 🔍 Related Issues

- vLLM expert-parallel routing replay for the TRTLLM-GEN fused MoE path

## 🚀 Pull Request Checklist

Thank you for contributing to FlashInfer! Before we review your pull request, please make sure the following items are complete.

### ✅ Pre-commit Checks

- [ ] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [ ] I have installed the hooks with `pre-commit install`.
- [ ] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure about how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

## 🧪 Tests

- [x] Tests have been added or updated as needed.
- [ ] All tests are passing (`unittest`, etc.).

**New tests added:**
- `test_routing_replay_out` — standalone `fused_topk_deepseek` routing kernel: verifies `routing_replay_out` matches `topk_indices` per token and that `None` produces identical results (no side effects)
- `test_fp8_block_scale_moe_routing_replay` — end-to-end FP8 block-scale MoE: verifies replay IDs are valid, unique per token, and MoE output is bit-identical with/without replay (SM100+)

## Reviewer Notes

- All changes are backward-compatible: `routing_replay_out` defaults to `None`, so existing callers are unaffected.
- The overhead when `routing_replay_out is None` is a single pointer null-check per thread — effectively zero.
- The write when enabled is K coalesced int16 stores per token (e.g., 8 stores for DeepSeek-V3's top-8), negligible relative to the routing computation.
