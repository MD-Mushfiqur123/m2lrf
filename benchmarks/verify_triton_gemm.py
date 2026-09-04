"""
M-2LRF Triton In-SRAM GEMM Verification Suite (Google Colab / GPU Target)
========================================================================
Validates:
  1. Numerical equivalence between Triton Fused GEMM and PyTorch Dequant Fallback
  2. Sub-tile bit-unpacking correctness across diverse matrix dimensions (M, N, K)
  3. Latency & throughput benchmarking on NVIDIA Tensor Cores (T4 / A100 / H100 / RTX 3090)
"""

import sys
import math
import time
from pathlib import Path

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F

from m2lrf.packed_codec import Real2BitCodec
from m2lrf.triton_kernel import HAS_TRITON, m2lrf_triton_matmul, m2lrf_matmul_fallback


def run_triton_numerical_verification():
    print("=" * 80)
    print("🚀 M-2LRF TRITON IN-SRAM GEMM NUMERICAL VERIFICATION SUITE")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[!] CUDA GPU is not detected in current environment.")
        print("[*] Running verification in PyTorch Fallback mode.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"[*] GPU Device Detected: {torch.cuda.get_device_name(device)}")
        print(f"[*] Triton Available   : {HAS_TRITON}")

    test_shapes = [
        # (M, N, K)
        (1, 4096, 4096),     # Single-token decode step
        (4, 4096, 4096),     # Small batch inference
        (16, 4096, 4096),    # Medium batch inference
        (128, 4096, 4096),   # Sequence prefill / fine-tuning step
        (4, 11008, 4096),    # LLaMA / Qwen MLP Intermediate Projection (Gate / Up)
        (4, 4096, 11008),    # LLaMA / Qwen MLP Down Projection
    ]

    all_passed = True

    print("\n" + "-" * 80)
    print(f"{'Shape (M, N, K)':<22} | {'Max Abs Diff':<15} | {'Rel Diff':<12} | {'Status'}")
    print("-" * 80)

    for M, N, K in test_shapes:
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.float16, device=device)
        w_orig = torch.randn(N, K, dtype=torch.float16, device=device)

        # Pack 2-bit weights
        packed_bytes, a0, a1, orig_shape = Real2BitCodec.pack(w_orig)

        # 1. PyTorch Dequant Fallback Output
        out_fallback = m2lrf_matmul_fallback(x, packed_bytes, a0, a1, orig_shape)

        # 2. Triton In-SRAM Fused Kernel Output
        out_triton = m2lrf_triton_matmul(x, packed_bytes, a0, a1, orig_shape)

        # Compute numerical divergence
        max_abs_diff = (out_triton.float() - out_fallback.float()).abs().max().item()
        rel_diff = (torch.norm(out_triton.float() - out_fallback.float()) / torch.norm(out_fallback.float())).item()

        # Tolerance check (allowing FP16 rounding differences on Tensor Cores)
        passed = (max_abs_diff < 0.05) and (rel_diff < 0.01)
        if not passed:
            all_passed = False

        status_str = "✅ PASS" if passed else "❌ FAIL"
        shape_str = f"({M}, {N}, {K})"
        print(f"{shape_str:<22} | {max_abs_diff:<15.6f} | {rel_diff:<12.6f} | {status_str}")

    print("-" * 80)
    if all_passed:
        print("🎉 ALL NUMERICAL EQUIVALENCE TESTS PASSED PERFECTLY!")
    else:
        print("⚠️ Some shape configurations exceeded numerical tolerance.")
    print("=" * 80 + "\n")


def run_latency_microbenchmark(M: int = 128, N: int = 4096, K: int = 4096, warmup: int = 20, trials: int = 100):
    if not torch.cuda.is_available():
        print("[!] GPU is required for latency microbenchmarking.")
        return

    device = torch.device("cuda:0")
    print("=" * 80)
    print(f"⚡ LATENCY & THROUGHPUT BENCHMARK (Shape: M={M}, N={N}, K={K})")
    print("=" * 80)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w_orig = torch.randn(N, K, dtype=torch.float16, device=device)
    packed_bytes, a0, a1, orig_shape = Real2BitCodec.pack(w_orig)

    # 1. Standard PyTorch FP16 Linear
    for _ in range(warmup):
        _ = F.linear(x, w_orig)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = F.linear(x, w_orig)
    torch.cuda.synchronize()
    lat_fp16 = ((time.perf_counter() - t0) / trials) * 1000.0

    # 2. PyTorch Fallback Dequant + GEMM
    for _ in range(warmup):
        _ = m2lrf_matmul_fallback(x, packed_bytes, a0, a1, orig_shape)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = m2lrf_matmul_fallback(x, packed_bytes, a0, a1, orig_shape)
    torch.cuda.synchronize()
    lat_fallback = ((time.perf_counter() - t0) / trials) * 1000.0

    # 3. Triton In-SRAM Fused GEMM
    for _ in range(warmup):
        _ = m2lrf_triton_matmul(x, packed_bytes, a0, a1, orig_shape)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = m2lrf_triton_matmul(x, packed_bytes, a0, a1, orig_shape)
    torch.cuda.synchronize()
    lat_triton = ((time.perf_counter() - t0) / trials) * 1000.0

    print(f"  [+] Unquantized FP16 Baseline GEMM : {lat_fp16:.3f} ms")
    print(f"  [+] PyTorch Dequant + GEMM Fallback: {lat_fallback:.3f} ms")
    print(f"  [+] M-2LRF Fused In-SRAM Triton GEMM: {lat_triton:.3f} ms")
    if lat_triton < lat_fallback:
        print(f"  [+] Triton Speedup over Fallback   : {lat_fallback / lat_triton:.2f}x")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_triton_numerical_verification()
    if torch.cuda.is_available():
        run_latency_microbenchmark()
