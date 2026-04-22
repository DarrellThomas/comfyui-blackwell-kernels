#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

"""
A/B Performance Test — Stock PyTorch vs Blackwell Kernels

Run this to see the speed difference on your RTX 5090:
    python benchmark.py

No ComfyUI needed — this is a standalone microbenchmark.
"""

import sys
import torch
import torch.nn.functional as F

# ── Check GPU ──

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected.")
    sys.exit(1)

cc = torch.cuda.get_device_capability()
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name} (Compute Capability {cc[0]}.{cc[1]})")
print()

if cc < (12, 0):
    print(f"WARNING: This GPU is CC {cc[0]}.{cc[1]}. Blackwell kernels need CC 12.0+ (RTX 5090).")
    print("Stock PyTorch benchmark will still run.\n")

# ── Try loading custom kernel ──

has_blackwell = False
try:
    from blackwell_kernels.attention import flash_attention
    has_blackwell = True
    print("[OK] Blackwell kernels loaded successfully.")
except ImportError as e:
    print(f"[!!] Blackwell kernels not compiled: {e}")
    print("     Only stock PyTorch will be benchmarked.")
    print("     Build with: CUDA_HOME=/usr/local/cuda-13 pip install -e .")
print()

# ── Benchmark configs ──
# These match real ComfyUI workloads (SD 1.5, SDXL, Flux)

CONFIGS = [
    # (name, B, H, N, D, causal)
    ("SD 1.5 self-attn (D=40)",     4, 32, 1024, 40,  False),
    ("SD 1.5 cross-attn (D=64)",    4, 32, 1024, 64,  False),
    ("SDXL self-attn (D=64)",       2, 16, 2048, 64,  False),
    ("SDXL cross-attn (D=128)",     2, 16, 2048, 128, False),
    ("Flux (D=128, long seq)",       1,  8, 4096, 128, False),
]

WARMUP = 20
ITERS = 100
device = "cuda:0"


def bench_stock(q, k, v, warmup=WARMUP, iters=ITERS):
    """Benchmark stock PyTorch SDPA (cuDNN backend)."""
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def bench_blackwell(q, k, v, warmup=WARMUP, iters=ITERS):
    """Benchmark Blackwell custom kernel."""
    for _ in range(warmup):
        flash_attention(q, k, v, causal=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        flash_attention(q, k, v, causal=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ── Run benchmarks ──

print("=" * 78)
print(f"{'Config':<32} {'Stock (ms)':>10} {'Blackwell (ms)':>14} {'Speedup':>10}")
print("-" * 78)

for name, B, H, N, D, causal in CONFIGS:
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16)

    stock_ms = bench_stock(q, k, v)

    if has_blackwell and cc >= (12, 0):
        bk_ms = bench_blackwell(q, k, v)
        speedup = stock_ms / bk_ms
        tag = "FASTER" if speedup > 1.02 else ("same" if speedup > 0.98 else "slower")
        print(f"{name:<32} {stock_ms:>10.3f} {bk_ms:>14.3f} {speedup:>8.2f}x  {tag}")
    else:
        print(f"{name:<32} {stock_ms:>10.3f} {'n/a':>14} {'n/a':>10}")

    del q, k, v
    torch.cuda.empty_cache()

print("=" * 78)
print()

if has_blackwell and cc >= (12, 0):
    print("Blackwell kernels are working. When ComfyUI runs, every attention")
    print("operation uses these faster kernels automatically.")
else:
    print("Build the kernels to see the comparison:")
    print("  CUDA_HOME=/usr/local/cuda-13 pip install -e .")
