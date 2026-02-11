#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for Nunchaku layernorm Kernel.

Profiles the layernorm kernel with L2 cache clearing to get accurate
memory bandwidth measurements. The kernel performs:
    out = (input - mean) / sqrt(var + eps) * weight + bias

Usage:
    # Basic run with default params (1024 tokens, 3072 hidden dim)
    python scripts/profile_layernorm_kernel_bandwidth.py

    # Custom parameters
    python scripts/profile_layernorm_kernel_bandwidth.py --tokens 4096 --hidden-dim 4096

    # Without L2 cache clearing (for comparison)
    python scripts/profile_layernorm_kernel_bandwidth.py --no-clear-cache

    # With nsys profiling
    nsys profile -o layernorm_kernel python scripts/profile_layernorm_kernel_bandwidth.py
"""

import argparse
import statistics
from dataclasses import dataclass

import torch

from nunchaku._C import ops
from nunchaku.profiling.bandwidth_profiler import GPU_PEAK_BANDWIDTH


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
class LayernormKernelBytes:
    """Memory traffic breakdown for layernorm kernel.

    The layernorm kernel performs: out = (input - mean) / sqrt(var + eps) * weight + bias
    - Reads: input tensor, weight (broadcasted), bias (broadcasted)
    - Writes: output tensor

    Note: weight and bias are [hidden_dim] and broadcasted across [tokens, hidden_dim].
    For large token counts, they are L2-cached after first access, so we only count
    them once in the memory traffic.

    The kernel uses shared memory for intermediate values (mean/variance computation),
    so input is only read once from global memory.
    """

    input_bytes: int  # tokens x hidden_dim x dtype_size (read input)
    weight_bytes: int  # hidden_dim x dtype_size (read weight, L2 cached)
    bias_bytes: int  # hidden_dim x dtype_size (read bias, L2 cached)
    output_bytes: int  # tokens x hidden_dim x dtype_size (write output)

    @property
    def total_read_bytes(self) -> int:
        return self.input_bytes + self.weight_bytes + self.bias_bytes

    @property
    def total_write_bytes(self) -> int:
        return self.output_bytes

    @property
    def total_bytes(self) -> int:
        return self.total_read_bytes + self.total_write_bytes


def calculate_layernorm_bytes(
    tokens: int, hidden_dim: int, dtype_size: int = 2
) -> LayernormKernelBytes:
    """Calculate memory traffic for layernorm kernel.

    Args:
        tokens: Number of tokens (M dimension)
        hidden_dim: Hidden dimension (K dimension)
        dtype_size: Size of each element in bytes (2 for bf16)

    Returns:
        LayernormKernelBytes with detailed memory traffic breakdown
    """
    tensor_bytes = tokens * hidden_dim * dtype_size
    weight_bias_bytes = hidden_dim * dtype_size

    return LayernormKernelBytes(
        input_bytes=tensor_bytes,
        weight_bytes=weight_bias_bytes,
        bias_bytes=weight_bias_bytes,
        output_bytes=tensor_bytes,
    )


# --- Single Kernel Profile ---


@dataclass
class ProfileResult:
    """Results from kernel profiling."""

    times_ms: list[float]
    bytes_info: LayernormKernelBytes
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


def profile_layernorm_kernel(
    tokens: int,
    hidden_dim: int,
    warmup_iters: int = 5,
    profile_iters: int = 100,
    clear_cache: bool = True,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    dtype: torch.dtype = torch.bfloat16,
    epsilon: float = 1e-5,
) -> ProfileResult:
    """Profile the layernorm kernel.

    Args:
        tokens: Number of tokens (M dimension)
        hidden_dim: Hidden dimension (K dimension)
        warmup_iters: Number of warmup iterations (no timing)
        profile_iters: Number of profiled iterations
        clear_cache: Whether to clear L2 cache before each iteration
        peak_gbs: Peak GPU bandwidth for efficiency calculation
        dtype: Data type for tensors
        epsilon: Epsilon for numerical stability in layernorm

    Returns:
        ProfileResult with timing statistics and bandwidth calculations
    """
    dtype_size = torch.finfo(dtype).bits // 8

    # Allocate tensors
    # input/output are [tokens, hidden_dim], weight and bias are [hidden_dim]
    input_tensor = torch.randn(tokens, hidden_dim, dtype=dtype, device="cuda")
    output_tensor = torch.empty(tokens, hidden_dim, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    bias = torch.randn(hidden_dim, dtype=dtype, device="cuda")

    # Calculate expected bytes
    bytes_info = calculate_layernorm_bytes(
        tokens=tokens, hidden_dim=hidden_dim, dtype_size=dtype_size
    )

    # Warmup (without L2 clearing to let caches warm up)
    for _ in range(warmup_iters):
        ops.layernorm(output_tensor, input_tensor, weight, bias, epsilon)
    torch.cuda.synchronize()

    # Profile with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for _ in range(profile_iters):
        if clear_cache:
            clear_l2_cache()

        start.record()
        ops.layernorm(output_tensor, input_tensor, weight, bias, epsilon)
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
    clear_cache: bool,
    profile_iters: int,
    dtype: torch.dtype,
    epsilon: float,
):
    """Print formatted profiling report."""
    numel = tokens * hidden_dim
    dtype_size = torch.finfo(dtype).bits // 8
    cache_status = "L2 cache cleared" if clear_cache else "L2 cache warm"

    print("=" * 74)
    print("layernorm Kernel Memory Bandwidth Profile")
    print("=" * 74)
    print()
    print("Configuration:")
    print(f"  Tokens (M):      {tokens}")
    print(f"  Hidden dim (K):  {hidden_dim}")
    print(f"  Total elements:  {numel:,}")
    print(f"  Data type:       {dtype} ({dtype_size} bytes/element)")
    print(f"  Epsilon:         {epsilon}")
    print()
    print("Tensor Shapes:")
    print(f"  input:         [{tokens}, {hidden_dim}] {dtype}")
    print(f"  output:        [{tokens}, {hidden_dim}] {dtype}")
    print(f"  weight:        [{hidden_dim}] {dtype}")
    print(f"  bias:          [{hidden_dim}] {dtype}")
    print()
    print("Operation: out = (input - mean) / sqrt(var + eps) * weight + bias")
    print()
    print("Memory Traffic:")
    b = result.bytes_info
    print(
        f"  Read input:      {format_bytes(b.input_bytes):>12} (tokens x hidden_dim x {dtype_size})"
    )
    print(
        f"  Read weight:     {format_bytes(b.weight_bytes):>12} (hidden_dim x {dtype_size}, L2 cached)"
    )
    print(
        f"  Read bias:       {format_bytes(b.bias_bytes):>12} (hidden_dim x {dtype_size}, L2 cached)"
    )
    print(
        f"  Write output:    {format_bytes(b.output_bytes):>12} (tokens x hidden_dim x {dtype_size})"
    )
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
        description="Profile memory bandwidth for nunchaku layernorm kernel"
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
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="Epsilon for numerical stability",
    )

    args = parser.parse_args()

    # Override peak bandwidth if GPU specified
    if args.gpu:
        args.peak_bandwidth = GPU_PEAK_BANDWIDTH[args.gpu]

    clear_cache = not args.no_clear_cache

    # Effective tokens = batch_size * tokens
    effective_tokens = args.batch_size * args.tokens

    print(f"Profiling nunchaku layernorm kernel...")
    print(
        f"  Batch size: {args.batch_size}, Tokens: {args.tokens}, Effective tokens: {effective_tokens}"
    )
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Clear L2 cache: {clear_cache}")
    print()

    result = profile_layernorm_kernel(
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        warmup_iters=args.warmup,
        profile_iters=args.iters,
        clear_cache=clear_cache,
        peak_gbs=args.peak_bandwidth,
        epsilon=args.epsilon,
    )

    print_report(
        result=result,
        tokens=effective_tokens,
        hidden_dim=args.hidden_dim,
        clear_cache=clear_cache,
        profile_iters=args.iters,
        dtype=torch.bfloat16,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
