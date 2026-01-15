"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys

import flashinfer
import numpy as np
import torch
import triton
from flashinfer.testing.utils import bench_gpu_time, bench_gpu_time_with_cudagraph

# Add the project root to Python path to import test helpers
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.test_helpers.rope_reference import RotaryEmbedding

mode_ncu = bool(int(os.environ.get("FLASHINFER_MODE_NCU", "0")))


def benchmark_config(
    config_name, num_tokens, provider, batch_size=4, page_size=16, enable_pdl=False
):
    """Benchmark fused vs unfused rope_quantize_fp8 + append_paged_kv_cache.

    Args:
        config_name: "mla", "gqa", or "mha"
        num_tokens: Number of tokens to process
        provider: "fused", "unfused", "rope_only", or "append_only"
        batch_size: Batch size for paging (default: 4)
        page_size: Page size for KV cache (default: 16)
        enable_pdl: Enable PDL for rope_quantize (default: False)
    """
    input_dtype = torch.bfloat16
    device = "cuda"
    quant_dtype = torch.float8_e4m3fn

    # Configuration-specific parameters
    if config_name == "mla":
        # MLA: Original configuration for regression testing
        num_qo_heads, num_kv_heads = 128, 1
        rope_dim, no_rope_dim = 64, 512
    elif config_name == "gqa":
        # GQA: Realistic grouped-query attention
        num_qo_heads, num_kv_heads = 32, 8
        rope_dim, no_rope_dim = 64, 64
    elif config_name == "mha":
        # MHA: Standard multi-head attention
        num_qo_heads, num_kv_heads = 32, 32
        rope_dim, no_rope_dim = 64, 64
    else:
        raise ValueError(f"Unknown config: {config_name}")

    head_dim = rope_dim + no_rope_dim

    # Create input tensors
    if config_name == "mla":
        # MLA: 2D K tensors (shared)
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = torch.randn(
            num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        k_rope = torch.randn(num_tokens, rope_dim, dtype=input_dtype, device=device)
        k_nope = torch.randn(num_tokens, no_rope_dim, dtype=input_dtype, device=device)
        v = None
    else:
        # GQA/MHA: 3D K/V tensors
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = torch.randn(
            num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        k_rope = torch.randn(
            num_tokens, num_kv_heads, rope_dim, dtype=input_dtype, device=device
        )
        k_nope = torch.randn(
            num_tokens, num_kv_heads, no_rope_dim, dtype=input_dtype, device=device
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=input_dtype, device=device
        )

    # Create RoPE reference for cos/sin cache
    max_seq_len = max(4096, num_tokens)
    rope_ref = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rope_dim,
        max_position_embeddings=max_seq_len,
        base=10000,
        is_neox_style=False,
        dtype=input_dtype,
        device=device,
    )
    pos_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    # Build paged metadata (single request with all tokens)
    kv_append_length = torch.tensor(
        [num_tokens] + [0] * (batch_size - 1), dtype=torch.int32, device=device
    )
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )
    num_pages_per_req = torch.tensor(
        [(num_tokens + page_size - 1) // page_size] + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(
        kv_page_indptr[-1].item(), dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [num_tokens % page_size if num_tokens % page_size != 0 else page_size]
        + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )

    # Get batch_indices and positions
    seq_lens = flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr, seq_lens, num_tokens
    )

    # Allocate caches
    max_pages = kv_page_indptr[-1].item()

    if config_name == "mla":
        ckv_cache = torch.zeros(
            max_pages, page_size, no_rope_dim, dtype=quant_dtype, device=device
        )
        kpe_cache = torch.zeros(
            max_pages, page_size, rope_dim, dtype=quant_dtype, device=device
        )
        paged_kv_cache = (ckv_cache, kpe_cache)
    else:
        # GQA/MHA: use NHD layout
        k_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        v_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=quant_dtype,
            device=device,
        )
        paged_kv_cache = (k_cache, v_cache)

    run_idx = 0

    if provider == "fused":
        # Fused approach: single kernel call
        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                v=v,
                cos_sin_cache=rope_ref.cos_sin_cache,
                pos_ids=pos_ids,
                paged_kv_cache=paged_kv_cache,
                kv_indices=kv_page_indices,
                kv_indptr=kv_page_indptr,
                batch_indices=batch_indices,
                positions=positions,
                page_size=page_size,
                kv_layout="NHD",
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                is_neox=False,
                enable_pdl=enable_pdl,
            )

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    elif provider == "unfused":
        # Unfused approach: separate rope_quantize_fp8 + append calls
        # Allocate intermediate outputs for rope_quantize_fp8
        q_rope_out = torch.empty_like(q_rope, dtype=quant_dtype)
        k_rope_out = torch.empty_like(k_rope, dtype=quant_dtype)
        q_nope_out = torch.empty_like(q_nope, dtype=quant_dtype)
        k_nope_out = torch.empty_like(k_nope, dtype=quant_dtype)

        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            # Step 1: RoPE quantize
            flashinfer.rope.rope_quantize_fp8(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=rope_ref.cos_sin_cache,
                pos_ids=pos_ids,
                is_neox=False,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                q_rope_out=q_rope_out,
                k_rope_out=k_rope_out,
                q_nope_out=q_nope_out,
                k_nope_out=k_nope_out,
                enable_pdl=enable_pdl,
            )

            # Step 2: Append to paged KV cache
            if config_name == "mla":
                # MLA uses append_paged_mla_kv_cache
                flashinfer.page.append_paged_mla_kv_cache(
                    append_ckv=k_nope_out,
                    append_kpe=k_rope_out,
                    batch_indices=batch_indices,
                    positions=positions,
                    ckv_cache=paged_kv_cache[0],
                    kpe_cache=paged_kv_cache[1],
                    kv_indices=kv_page_indices,
                    kv_indptr=kv_page_indptr,
                    kv_last_page_len=kv_last_page_len,
                )
            else:
                # GQA/MHA: concatenate k_rope and k_nope, then use append_paged_kv_cache
                k_full = torch.cat([k_rope_out, k_nope_out], dim=-1)
                # Need to quantize v as well for GQA/MHA
                v_out = v.to(quant_dtype)

                flashinfer.page.append_paged_kv_cache(
                    append_key=k_full,
                    append_value=v_out,
                    batch_indices=batch_indices,
                    positions=positions,
                    paged_kv_cache=paged_kv_cache,
                    kv_indices=kv_page_indices,
                    kv_indptr=kv_page_indptr,
                    kv_last_page_len=kv_last_page_len,
                    kv_layout="NHD",
                )

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    elif provider == "rope_only":
        # Measure only rope_quantize_fp8
        q_rope_out = torch.empty_like(q_rope, dtype=quant_dtype)
        k_rope_out = torch.empty_like(k_rope, dtype=quant_dtype)
        q_nope_out = torch.empty_like(q_nope, dtype=quant_dtype)
        k_nope_out = torch.empty_like(k_nope, dtype=quant_dtype)

        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            flashinfer.rope.rope_quantize_fp8(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=rope_ref.cos_sin_cache,
                pos_ids=pos_ids,
                is_neox=False,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                q_rope_out=q_rope_out,
                k_rope_out=k_rope_out,
                q_nope_out=q_nope_out,
                k_nope_out=k_nope_out,
                enable_pdl=enable_pdl,
            )

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    elif provider == "append_only":
        # Measure append + any necessary preprocessing (to match unfused path)
        q_rope_out = torch.empty_like(q_rope, dtype=quant_dtype)
        k_rope_out = torch.empty_like(k_rope, dtype=quant_dtype)
        q_nope_out = torch.empty_like(q_nope, dtype=quant_dtype)
        k_nope_out = torch.empty_like(k_nope, dtype=quant_dtype)

        # Pre-compute once
        flashinfer.rope.rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=rope_ref.cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=quant_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            q_rope_out=q_rope_out,
            k_rope_out=k_rope_out,
            q_nope_out=q_nope_out,
            k_nope_out=k_nope_out,
            enable_pdl=enable_pdl,
        )

        def execute():
            nonlocal run_idx
            run_idx += 1

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStart()

            if config_name == "mla":
                flashinfer.page.append_paged_mla_kv_cache(
                    append_ckv=k_nope_out,
                    append_kpe=k_rope_out,
                    batch_indices=batch_indices,
                    positions=positions,
                    ckv_cache=paged_kv_cache[0],
                    kpe_cache=paged_kv_cache[1],
                    kv_indices=kv_page_indices,
                    kv_indptr=kv_page_indptr,
                    kv_last_page_len=kv_last_page_len,
                )
            else:
                # Include concat and type conversion to match unfused path
                k_full = torch.cat([k_rope_out, k_nope_out], dim=-1)
                v_out = v.to(quant_dtype)

                flashinfer.page.append_paged_kv_cache(
                    append_key=k_full,
                    append_value=v_out,
                    batch_indices=batch_indices,
                    positions=positions,
                    paged_kv_cache=paged_kv_cache,
                    kv_indices=kv_page_indices,
                    kv_indptr=kv_page_indptr,
                    kv_last_page_len=kv_last_page_len,
                    kv_layout="NHD",
                )

            if mode_ncu and run_idx == 20:
                torch.cuda.cudart().cudaProfilerStop()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    if mode_ncu:
        measurements = bench_gpu_time(execute)
    else:
        measurements = bench_gpu_time_with_cudagraph(execute, cold_l2_cache=False)
    # Calculate statistics
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


# Create separate benchmark functions for each architecture
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["fused", "unfused"],
        line_names=["Fused (single kernel)", "Unfused (rope + append)"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="Latency (ms)",
        plot_name="mla-fusion-benchmark",
        args={},
    )
)
def benchmark_mla(provider, num_tokens):
    return benchmark_config("mla", num_tokens, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["fused", "unfused"],
        line_names=["Fused (single kernel)", "Unfused (rope + append)"],
        styles=[("green", "-"), ("orange", "--")],
        ylabel="Latency (ms)",
        plot_name="gqa-fusion-benchmark",
        args={},
    )
)
def benchmark_gqa(provider, num_tokens):
    return benchmark_config("gqa", num_tokens, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[768] if mode_ncu else [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768],
        line_arg="provider",
        line_vals=["fused", "unfused"],
        line_names=["Fused (single kernel)", "Unfused (rope + append)"],
        styles=[("purple", "-"), ("brown", "--")],
        ylabel="Latency (ms)",
        plot_name="mha-fusion-benchmark",
        args={},
    )
)
def benchmark_mha(provider, num_tokens):
    return benchmark_config("mha", num_tokens, provider)


if __name__ == "__main__":
    # Run all benchmarks and generate individual plots
    print("Running MLA fusion benchmark...")
    benchmark_mla.run(print_data=False, show_plots=True, save_path=".")

    print("Running GQA fusion benchmark...")
    benchmark_gqa.run(print_data=False, show_plots=True, save_path=".")

    print("Running MHA fusion benchmark...")
    benchmark_mha.run(print_data=False, show_plots=True, save_path=".")

    # Collect results for summary tables
    token_counts = (
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768] if not mode_ncu else [768]
    )

    # Helper function to print a table for one config
    def print_config_table(config_name, token_counts):
        print("\n" + "=" * 100)
        print(f"=== {config_name.upper()} Results ===")
        print("=" * 100)
        print(
            f"{'Tokens':<8} {'Fused':<12} {'Unfused':<12} {'RoPE+Quant%':<13} "
            f"{'Append%':<10} {'Speedup':<12}"
        )
        print("-" * 100)

        for num_tokens in token_counts:
            fused_ms, _, _ = benchmark_config(config_name, num_tokens, "fused")
            unfused_ms, _, _ = benchmark_config(config_name, num_tokens, "unfused")
            rope_ms, _, _ = benchmark_config(config_name, num_tokens, "rope_only")
            append_ms, _, _ = benchmark_config(config_name, num_tokens, "append_only")
            speedup = unfused_ms / fused_ms

            # Calculate percentages
            rope_pct = (rope_ms / unfused_ms) * 100
            append_pct = (append_ms / unfused_ms) * 100

            print(
                f"{num_tokens:<8} "
                f"{fused_ms:<12.5f} {unfused_ms:<12.5f} {rope_pct:>11.1f}%  "
                f"{append_pct:>8.1f}%  {speedup:>8.3f}x"
            )

        print("=" * 100)

    # Print separate tables for each configuration
    print_config_table("mla", token_counts)
    print_config_table("gqa", token_counts)
    print_config_table("mha", token_counts)

    print("Configuration details:")
    print("  MLA: 128 Q heads, 1 K head, 64+512 dims (DeepSeek-style)")
    print("  GQA: 32 Q heads, 8 K heads, 64+64 dims (Llama-style)")
    print("  MHA: 32 Q heads, 32 K heads, 64+64 dims (Standard)")
    print("  Page size: 16, Batch size: 4")
    print("\nMethods:")
    print("  Fused: rope_quantize_fp8_append_paged_kv_cache (single kernel)")
    print(
        "  Unfused: rope_quantize_fp8 + append_paged_mla_kv_cache/append_paged_kv_cache (separate)"
    )
    print("  RoPE+Quant%: Percentage of unfused time spent in rope_quantize_fp8")
    print("  Append%: Percentage of unfused time spent in append_paged_*_kv_cache")

    print("\nPlot files saved to current directory:")
    print("  mla-fusion-benchmark.png")
    print("  gqa-fusion-benchmark.png")
    print("  mha-fusion-benchmark.png")
    print("=" * 100)
