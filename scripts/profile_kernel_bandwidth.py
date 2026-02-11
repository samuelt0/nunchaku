#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for Individual Nunchaku Kernels.

Profiles individual CUDA kernels with L2 cache clearing to get accurate
memory bandwidth measurements, starting with the quantize_w4a4_fuse_lora_kernel.

Usage:
    # Basic run with default params (4096 tokens, 4096 hidden dim)
    python scripts/profile_kernel_bandwidth.py

    # With FP4 quantization
    python scripts/profile_kernel_bandwidth.py --fp4

    # Custom parameters
    python scripts/profile_kernel_bandwidth.py --tokens 4096 --hidden-dim 4096 --lora-rank 64

    # Without L2 cache clearing (for comparison)
    python scripts/profile_kernel_bandwidth.py --no-clear-cache

    # With nsys profiling
    nsys profile -o quantize_kernel python scripts/profile_kernel_bandwidth.py
"""

import argparse
import statistics
from dataclasses import dataclass

import torch

from nunchaku._C import ops
from nunchaku.profiling.bandwidth_profiler import GPU_PEAK_BANDWIDTH
from nunchaku.utils import ceil_divide


# --- L2 Cache Clearing ---


def clear_l2_cache(size_mb: int = 128):
    """Clear L2 cache by allocating and memset'ing a large buffer.

    This technique is from NVIDIA nvbench. By allocating a buffer larger than
    the L2 cache and filling it with zeros, we evict any cached data from
    previous kernel runs, ensuring the next kernel starts with a cold cache.

    Args:
        size_mb: Size of scratch buffer in MB. B200 has ~96MB L2, so 128MB
            is used by default to ensure full eviction.
    """
    size_bytes = size_mb * 1024 * 1024
    scratch = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    scratch.zero_()  # cudaMemset fills L2 with this data, evicting old cached data
    torch.cuda.synchronize()
    del scratch


# --- Bandwidth Calculation ---


@dataclass
class QuantizeKernelBytes:
    """Memory traffic breakdown for quantize_w4a4_fuse_lora kernel.

    Based on actual kernel implementation in src/kernels/zgemm/gemm_w4a4.cuh:
    - Kernel reads input activations and smooth factor
    - Kernel writes quantized output and output scales
    - NOTE: LoRA computation is currently DISABLED in the kernel (commented out)
      so lora_down is not read and lora_act_out is not written by the kernel,
      though the tensors are still allocated/passed by Python.

    Memory layout (from gemm_base.cuh and gemm_w4a4.cuh):
    - BLOCK_M = 256, BLOCK_N = 128, WARP_K = 64
    - Input is read in tiles of WARP_M x WARP_N per warp
    - Output is written as packed int4 (2 values per uint8)
    - Scales: FP4 uses group_size=16, INT4 uses group_size=64
    """

    input_bytes: int  # actualM x actualN x 2 (bf16 input)
    smooth_bytes: int  # K x 2 (bf16 smooth factor, if provided) - may be L2 cached
    output_bytes: int  # M_pad x K/2 (packed int4 as uint8)
    oscales_bytes: int  # M_pad x (K/group_size) x dtype_size

    @property
    def total_read_bytes(self) -> int:
        return self.input_bytes + self.smooth_bytes

    @property
    def total_write_bytes(self) -> int:
        return self.output_bytes + self.oscales_bytes

    @property
    def total_bytes(self) -> int:
        return self.total_read_bytes + self.total_write_bytes


def calculate_quantize_bytes(
    M: int,
    K: int,
    fp4: bool = False,
    has_smooth: bool = True,
    pad_size: int = 256,
) -> QuantizeKernelBytes:
    """Calculate memory traffic for quantize_w4a4_fuse_lora kernel.

    Based on actual kernel code in src/kernels/zgemm/gemm_w4a4.cuh:
    - quantize_w4a4_fuse_lora_kernel::operator() (line 1125)
    - EpilogueQuantize::apply_quantize (line 945)

    Memory access pattern:
    - Reads: input tensor (M x K x bf16), smooth_factor (K x bf16)
    - Writes: output (M_pad x K/2 x uint8), oscales (M_pad x K/G x dtype)

    NOTE: LoRA is disabled in current kernel (TODO comment at line 1160).

    Args:
        M: Number of tokens (actualM in kernel)
        K: Hidden dimension (actualN in kernel, called K here for clarity)
        fp4: Whether using FP4 (group_size=16) or INT4 (group_size=64)
        has_smooth: Whether smooth factor is provided
        pad_size: Padding for M dimension (BLOCK_M = 256 in kernel)

    Returns:
        QuantizeKernelBytes with detailed memory traffic breakdown
    """
    # Pad M to BLOCK_M (256) boundary
    BLOCK_M = 256
    M_pad = ceil_divide(M, BLOCK_M) * BLOCK_M

    # Group size for quantization scales
    # FP4: WARP_K=64, but scales are per 16 elements (from amscale layout)
    # INT4: WARP_K=64, scales are per 64 elements (from ascale layout)
    group_size = 16 if fp4 else 64

    # Input: M x K x 2 bytes (bf16)
    # Kernel reads actualM x actualN elements via load_act_to_fpsum
    input_bytes = M * K * 2

    # Smooth factor: K x 2 bytes (bf16)
    # Loaded via load_wscale() in EpilogueQuantize, likely L2 cached for small K
    smooth_bytes = K * 2 if has_smooth else 0

    # Output: M_pad x K/2 bytes (packed int4 as uint8)
    # Written via store() in EpilogueQuantize::apply_quantize (line 1003)
    output_bytes = M_pad * (K // 2)

    # Output scales: M_pad x (K / group_size) x dtype_size
    # FP4: float8_e4m3fn (1 byte per scale), stored as packed_amscale_t
    # INT4: bf16 (2 bytes per scale), stored as packed_ascale_t
    # From launch code (line 472-476):
    #   FP4: oscales.numel() == M * N / WARP_K * 4 = M * K / 64 * 4 = M * K / 16
    #   INT4: oscales.numel() == M * N / WARP_K = M * K / 64
    oscales_dtype_size = 1 if fp4 else 2
    num_scale_groups = K // group_size
    oscales_bytes = M_pad * num_scale_groups * oscales_dtype_size

    return QuantizeKernelBytes(
        input_bytes=input_bytes,
        smooth_bytes=smooth_bytes,
        output_bytes=output_bytes,
        oscales_bytes=oscales_bytes,
    )


# --- Single Kernel Profile ---


@dataclass
class ProfileResult:
    """Results from kernel profiling."""

    times_ms: list[float]
    bytes_info: QuantizeKernelBytes
    peak_gbs: float

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def bandwidth_gbs(self) -> float:
        """Calculate bandwidth in GB/s from mean time."""
        time_s = self.mean_ms / 1000.0
        return self.bytes_info.total_bytes / time_s / 1e9 if time_s > 0 else 0.0

    @property
    def efficiency_pct(self) -> float:
        """Bandwidth efficiency as percentage of peak."""
        return (self.bandwidth_gbs / self.peak_gbs * 100) if self.peak_gbs > 0 else 0.0


def profile_quantize_kernel(
    tokens: int,
    hidden_dim: int,
    lora_rank: int = 64,
    fp4: bool = False,
    warmup_iters: int = 5,
    profile_iters: int = 100,
    clear_cache: bool = True,
    has_smooth: bool = True,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> ProfileResult:
    """Profile the quantize_w4a4_fuse_lora kernel.

    Args:
        tokens: Number of tokens (M dimension, called actualM in kernel)
        hidden_dim: Hidden dimension (K dimension, called actualN in kernel)
        lora_rank: LoRA rank (R dimension) - NOTE: LoRA is disabled in kernel
        fp4: Whether to use FP4 quantization (vs INT4)
        warmup_iters: Number of warmup iterations (no timing)
        profile_iters: Number of profiled iterations
        clear_cache: Whether to clear L2 cache before each iteration
        has_smooth: Whether to include smooth factor
        peak_gbs: Peak GPU bandwidth for efficiency calculation

    Returns:
        ProfileResult with timing statistics and bandwidth calculations
    """
    # Kernel constants from gemm_base.cuh
    BLOCK_M = 256
    BLOCK_N = 128

    M = tokens
    K = hidden_dim
    R = lora_rank
    M_pad = ceil_divide(M, BLOCK_M) * BLOCK_M
    group_size = 16 if fp4 else 64

    # Allocate input tensors
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    lora_down = torch.randn(K, R, dtype=torch.bfloat16, device="cuda")
    smooth = torch.randn(K, dtype=torch.bfloat16, device="cuda") if has_smooth else None

    # Allocate output tensors
    output = torch.empty(M_pad, K // 2, dtype=torch.uint8, device="cuda")
    if fp4:
        oscales = torch.empty(K // group_size, M_pad, dtype=torch.float8_e4m3fn, device="cuda")
    else:
        oscales = torch.empty(K // group_size, M_pad, dtype=torch.bfloat16, device="cuda")
    lora_act_out = torch.empty(M_pad, R, dtype=torch.float32, device="cuda")

    # Calculate expected bytes
    bytes_info = calculate_quantize_bytes(
        M=M,
        K=K,
        fp4=fp4,
        has_smooth=has_smooth,
        pad_size=BLOCK_M,
    )

    # Warmup (without L2 clearing to let caches warm up)
    fuse_glu = False
    for _ in range(warmup_iters):
        ops.quantize_w4a4_act_fuse_lora(
            input_tensor, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4
        )
    torch.cuda.synchronize()

    # Profile with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for _ in range(profile_iters):
        if clear_cache:
            clear_l2_cache()

        start.record()
        ops.quantize_w4a4_act_fuse_lora(
            input_tensor, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4
        )
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times_ms.append(elapsed_ms)

    return ProfileResult(times_ms=times_ms, bytes_info=bytes_info, peak_gbs=peak_gbs)


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.2f} KB"
    else:
        return f"{num_bytes} B"


def print_report(
    result: ProfileResult,
    tokens: int,
    hidden_dim: int,
    lora_rank: int,
    fp4: bool,
    clear_cache: bool,
    profile_iters: int,
    pad_size: int = 256,
):
    """Print formatted profiling report."""
    BLOCK_M = 256
    M_pad = ceil_divide(tokens, BLOCK_M) * BLOCK_M
    group_size = 16 if fp4 else 64
    precision = "FP4" if fp4 else "INT4"
    cache_status = "L2 cache cleared" if clear_cache else "L2 cache warm"

    print("=" * 74)
    print("Quantize Kernel Memory Bandwidth Profile")
    print("=" * 74)
    print()
    print("Configuration:")
    print(f"  Tokens (M):      {tokens}")
    print(f"  Hidden dim (K):  {hidden_dim}")
    print(f"  LoRA rank (R):   {lora_rank} (NOTE: LoRA disabled in kernel)")
    print(f"  Precision:       {precision} (group_size={group_size})")
    print(f"  BLOCK_M:         {BLOCK_M} -> M_pad = {M_pad}")
    print()
    print("Tensor Shapes:")
    print(f"  input:         [{tokens}, {hidden_dim}] bf16")
    print(f"  output:        [{M_pad}, {hidden_dim // 2}] uint8")
    print(f"  oscales:       [{hidden_dim // group_size}, {M_pad}] {'fp8_e4m3' if fp4 else 'bf16'}")
    print(f"  smooth:        [{hidden_dim}] bf16")
    print(f"  lora_down:     [{hidden_dim}, {lora_rank}] bf16 (not used by kernel)")
    print(f"  lora_act_out:  [{M_pad}, {lora_rank}] fp32 (not used by kernel)")
    print()
    print("Memory Traffic (from kernel analysis):")
    b = result.bytes_info
    print(f"  Input read:      {format_bytes(b.input_bytes):>12} (M x K x 2 bytes)")
    print(f"  Smooth read:     {format_bytes(b.smooth_bytes):>12} (K x 2 bytes)")
    print(f"  Output write:    {format_bytes(b.output_bytes):>12} (M_pad x K/2 bytes)")
    print(f"  Scales write:    {format_bytes(b.oscales_bytes):>12} (M_pad x K/{group_size} x {1 if fp4 else 2})")
    print(f"  ------------------------------------")
    print(f"  Total read:      {format_bytes(b.total_read_bytes):>12}")
    print(f"  Total write:     {format_bytes(b.total_write_bytes):>12}")
    print(f"  Total:           {format_bytes(b.total_bytes):>12}")
    print()
    print(f"Timing ({profile_iters} iterations, {cache_status}):")
    print(f"  Mean:  {result.mean_ms:.4f} ms")
    print(f"  Std:   {result.std_ms:.4f} ms")
    print(f"  Min:   {result.min_ms:.4f} ms")
    print(f"  Max:   {result.max_ms:.4f} ms")
    print()
    print("Bandwidth:")
    print(
        f"  {result.bandwidth_gbs:.1f} GB/s "
        f"({result.efficiency_pct:.1f}% of {result.peak_gbs:.0f} GB/s peak)"
    )
    print("=" * 74)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Profile memory bandwidth for quantize_w4a4_fuse_lora kernel"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (multiplies tokens)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1024,
        help="Number of tokens (M dimension)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=3072,
        help="Hidden dimension (K dimension)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank (R dimension)",
    )
    parser.add_argument(
        "--fp4",
        action="store_true",
        help="Use FP4 quantization (default: INT4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Disable L2 cache clearing (for comparison)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smooth factor",
    )
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=GPU_PEAK_BANDWIDTH["B200"],
        help="Peak GPU memory bandwidth in GB/s",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=list(GPU_PEAK_BANDWIDTH.keys()),
        help="Select GPU type for peak bandwidth (overrides --peak-bandwidth)",
    )

    args = parser.parse_args()

    # Override peak bandwidth if GPU specified
    if args.gpu:
        args.peak_bandwidth = GPU_PEAK_BANDWIDTH[args.gpu]

    clear_cache = not args.no_clear_cache
    has_smooth = not args.no_smooth

    # Effective tokens = batch_size * tokens
    effective_tokens = args.batch_size * args.tokens

    print(f"Profiling quantize_w4a4_fuse_lora kernel...")
    print(f"  Batch size: {args.batch_size}, Tokens: {args.tokens}, Effective tokens: {effective_tokens}")
    print(f"  Hidden dim: {args.hidden_dim}, LoRA rank: {args.lora_rank}")
    print(f"  FP4: {args.fp4}, Clear L2 cache: {clear_cache}")
    print()

    result = profile_quantize_kernel(
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        lora_rank=args.lora_rank,
        fp4=args.fp4,
        warmup_iters=args.warmup,
        profile_iters=args.iters,
        clear_cache=clear_cache,
        has_smooth=has_smooth,
        peak_gbs=args.peak_bandwidth,
    )

    print_report(
        result=result,
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        lora_rank=args.lora_rank,
        fp4=args.fp4,
        clear_cache=clear_cache,
        profile_iters=args.iters,
    )


if __name__ == "__main__":
    main()
