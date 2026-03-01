# Routing Replay: vLLM Integration Instructions

## Context

FlashInfer now supports an optional `routing_replay_out` parameter on `trtllm_fp8_block_scale_moe()`. When provided, the CUDA routing kernel writes all top-K selected expert IDs per token directly into this tensor during routing — inside the same fused kernel call that computes the MoE output.

**Why this is needed:** vLLM's TRTLLM-GEN fused MoE path is "monolithic" — `Fp8MoEMethod.apply_monolithic()` calls the FlashInfer kernel directly, **bypassing the router's `select_experts()`**. This means `BaseRouter.capture_fn` is never invoked, and the existing `RoutedExpertsCapturer` never receives expert IDs from this code path.

## FlashInfer API Change

```python
flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
    routing_logits=...,
    routing_bias=...,
    hidden_states=...,
    # ... all existing params unchanged ...
    routing_replay_out=None,  # NEW optional parameter
)
```

**`routing_replay_out`** spec:
- Type: `torch.Tensor` or `None`
- Dtype: `torch.int16`
- Shape: `[num_tokens, top_k]`
- Layout: row-major. `replay[t, k]` = the k-th ranked expert ID for token `t` (k=0 is highest score)
- When `None`: zero overhead, the kernel skips the write entirely
- When provided: the kernel writes expert IDs during routing with negligible overhead (K int16 stores per token, coalesced)

All existing parameters and behavior are unchanged. The parameter is at the end with a default of `None`, so all existing callers work without modification.

## What You Need to Change in vLLM

There are exactly 3 files to modify, each requiring a small change.

### File 1: `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`

**Function:** `flashinfer_fused_moe_blockscale_fp8()` (line ~186)

**What to do:** Add `routing_replay_out` parameter and pass it through.

```python
def flashinfer_fused_moe_blockscale_fp8(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    x: torch.Tensor,
    # ... existing params ...
    routed_scaling: float | None = 1.0,
    routing_replay_out: torch.Tensor | None = None,  # ADD THIS
) -> torch.Tensor:
    # ... existing code ...
    return flashinfer_trtllm_fp8_block_scale_moe(
        # ... existing kwargs ...
        routing_method_type=routing_method_type,
        use_shuffled_weight=False,
        routing_replay_out=routing_replay_out,  # ADD THIS
    )
```

Also update the corresponding `flashinfer_fused_moe_blockscale_fp8_fake` function (used by `torch.compile`) with the same parameter (it can be ignored in the fake implementation).

Also update the `@torch.library.custom_op` registration for `flashinfer_fused_moe_blockscale_fp8` to include `routing_replay_out` in its `mutates_args`.

### File 2: `vllm/model_executor/layers/quantization/fp8.py`

**Function:** `Fp8MoEMethod.apply_monolithic()` (line ~990)

**What to do:** Get the replay buffer from the `RoutedExpertsCapturer` device cache and pass it.

The `layer` parameter is a `FusedMoE` instance. You need to know the MoE layer index. The layer already assigns this in its `__init__` (see `FusedMoE._next_moe_layer_id` at line ~551 of `layer.py`), but it's stored only in the `capture_fn` closure. You'll need to also store it as an attribute on the layer.

```python
def apply_monolithic(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # ... existing assertions ...

    # Get routing replay buffer if capturer is active
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        get_global_experts_capturer,
    )
    routing_replay_out = None
    capturer = get_global_experts_capturer()
    device_cache = capturer.get_device_cache()
    if device_cache is not None and hasattr(layer, 'moe_layer_id'):
        num_tokens = x.shape[0]
        # device_cache.buffer: [max_tokens, num_layers, top_k]
        # Slice for this layer: [num_tokens, top_k] — contiguous because top_k is last dim
        routing_replay_out = device_cache.buffer[:num_tokens, layer.moe_layer_id, :]

    if self.block_quant:
        # ... existing import ...
        return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
            # ... existing kwargs ...
            routed_scaling=layer.routed_scaling_factor,
            routing_replay_out=routing_replay_out,  # ADD THIS
        )
    # ...
```

### File 3: `vllm/model_executor/layers/fused_moe/layer.py`

**Where:** `FusedMoE.__init__()`, near line ~551 where `moe_layer_id` is assigned.

**What to do:** Store the layer ID as an instance attribute so `apply_monolithic` can access it.

```python
moe_layer_id = FusedMoE._next_moe_layer_id
FusedMoE._next_moe_layer_id += 1
self.moe_layer_id = moe_layer_id  # ADD THIS LINE
self.router.set_capture_fn(
    lambda topk_ids, _lid=moe_layer_id:
        get_global_experts_capturer().capture(
            layer_id=_lid, topk_ids=topk_ids
        )
)
```

## Memory Layout Compatibility

vLLM's `_RoutedExpertsDeviceCache` buffer already uses:
- dtype: `torch.int16` — matches what FlashInfer writes
- shape: `[num_batched_tokens, num_hidden_layers, num_experts_per_tok]`

The slice `buffer[:num_tokens, layer_id, :]` yields a `[num_tokens, top_k]` view with `int16` dtype — exactly what FlashInfer expects. The view is contiguous because `top_k` is the innermost dimension.

## What NOT to Change

- **Do not** modify the router (`BaseRouter`, `GroupedTopKRouter`). The monolithic path bypasses the router entirely.
- **Do not** modify `RoutedExpertsCapturer` or its `capture()` method. The kernel writes directly to the device cache buffer, so the capturer's existing `sync_fwd_experts_buffer_DtoH` pipeline picks up the data automatically.
- **Do not** allocate any new tensors. The existing device cache buffer is reused.

## Verification

After integration, the flow is:

1. `DefaultMoERunner.forward()` calls `quant_method.apply_monolithic(layer, x, router_logits)`
2. `apply_monolithic` slices `device_cache.buffer[:N, layer_id, :]` → `[N, top_k]` int16 tensor
3. Passes it as `routing_replay_out` to FlashInfer's `trtllm_fp8_block_scale_moe`
4. The CUDA routing kernel writes expert IDs directly into that buffer slice
5. After the forward pass, `sync_fwd_experts_buffer_DtoH` copies the buffer D→H as before
6. `get_routed_experts()` extracts per-request data from the host cache as before

To verify correctness: for a non-fused path (non-monolithic), the router's `capture_fn` writes `topk_ids` into the same `device_cache.buffer`. Compare the values from both paths for the same input — they should match.
