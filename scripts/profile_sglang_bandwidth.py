#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for SGLang Element-wise Kernels (Llama 3.1 8B).

Calculates memory bandwidth utilization (GB/s) for element-wise kernels during
SGLang prefill, comparing effective bandwidth to B200's theoretical peak.

Usage:
    # SGLang uses flashinfer kernels for residual + layernorm, so use:
    python scripts/profile_sglang_bandwidth.py \
        --quantize-time 0.002484 --quantize-calls 256 \
        --fused-add-rms-norm-time 0.001945 --fused-add-rms-norm-calls 128 \
        --layernorm-time 0.000020992 --layernorm-calls 2 \
        --silu-time 0.004576 --silu-calls 64

    # Different model configuration
    python scripts/profile_sglang_bandwidth.py \
        --tokens 8192 \
        --hidden-dim 4096 \
        --intermediate-size 14336 \
        --num-layers 32 \
        --quantize-time 0.004 ...
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Common GPU peak memory bandwidth values (GB/s)
GPU_PEAK_BANDWIDTH = {
    "B200": 8000,  # ~8 TB/s
    "H100_SXM": 3350,
    "H100_PCIe": 2000,
    "A100_SXM": 2039,
    "A100_PCIe": 1935,
    "RTX_4090": 1008,
    "RTX_3090": 936,
}


@dataclass
class LlamaModelConfig:
    """Configuration for Llama model architecture.

    Attributes:
        hidden_dim: Model hidden dimension (d_model)
        intermediate_size: FFN intermediate dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per attention head
        tokens: Number of input tokens (prefill sequence length)
        dtype_bytes: Bytes per element (2 for bf16/fp16)
    """

    hidden_dim: int = 4096
    intermediate_size: int = 14336
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    tokens: int = 4096
    dtype_bytes: int = 2  # bf16


@dataclass
class KernelBandwidth:
    """Bandwidth results for a specific kernel type.

    Attributes:
        name: Kernel name (e.g., "cvt_fp16_to_fp4", "RMSNormKernel")
        time_s: Total execution time in seconds
        total_bytes: Total bytes transferred (read + write)
        bandwidth_gbs: Achieved bandwidth in GB/s
        efficiency_pct: Bandwidth efficiency as percentage of peak
        num_calls: Number of kernel invocations
        bytes_per_call: Bytes per single kernel invocation
    """

    name: str
    time_s: float
    total_bytes: int
    bandwidth_gbs: float
    efficiency_pct: float
    num_calls: int = 0
    bytes_per_call: int = 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_gb = self.total_bytes / 1e9
        return (
            f"{self.name:35s} {self.time_s:.6f} s | {size_gb:.4f} GB | "
            f"{self.bandwidth_gbs:.0f} GB/s | {self.efficiency_pct:.1f}%"
        )


def count_llama_kernel_calls(config: LlamaModelConfig) -> Dict[str, int]:
    """Count kernel calls for a single Llama prefill pass.

    Per-layer operations in Llama decoder:
    1. Pre-attention: FusedAddRMSNormKernel (residual + RMSNorm) -> 1 call
    2. Attention QKV projection: quantize before GEMM -> 1 call (fused QKV)
    3. Attention output projection: quantize before GEMM -> 1 call
    4. Pre-FFN: FusedAddRMSNormKernel (residual + RMSNorm) -> 1 call
    5. FFN gate/up projection: quantize before GEMM -> 1 call
    6. SiLU activation (SwiGLU): 1 call
    7. FFN down projection: quantize before GEMM -> 1 call

    Note: SGLang uses flashinfer's FusedAddRMSNormKernel for residual add + RMSNorm.
    The standalone RMSNormKernel is only used for the final norm after all layers.

    Args:
        config: LlamaModelConfig with model architecture

    Returns:
        Dict with kernel call counts: quantize, fused_add_rms_norm, rms_norm, silu
    """
    return {
        "quantize": config.num_layers * 4,  # QKV, out, gate/up, down per layer
        "fused_add_rms_norm": config.num_layers * 2,  # pre-attn + pre-FFN (fused residual + norm)
        "rms_norm": 1,  # final norm only (standalone)
        "silu": config.num_layers,  # SwiGLU activation per layer
    }


def calculate_llama_kernel_bytes(config: LlamaModelConfig) -> Dict[str, Dict]:
    """Calculate bytes for each kernel type in Llama.

    Memory access patterns:

    1. cvt_fp16_to_fp4 (Quantize):
       - Reads: input (BF16) = tokens * hidden_dim * 2 bytes
       - Writes: output (FP4 packed) = tokens * hidden_dim / 2 bytes, plus scales
       - Total: ~2.5 * tokens * hidden_dim bytes per call

    2. FusedAddRMSNormKernel (flashinfer fused residual + norm):
       - Reads: x (BF16) + residual (BF16) = 2 * tokens * hidden_dim * 2 bytes
       - Writes: output (BF16) = tokens * hidden_dim * 2 bytes
       - Weight/scale is tiny and cached in L2
       - Total: 3 * tokens * hidden_dim * 2 bytes per call

    3. RMSNormKernel (standalone, final norm only):
       - Reads: input (BF16) = tokens * hidden_dim * 2 bytes
       - Writes: output (BF16) = tokens * hidden_dim * 2 bytes
       - scale is small and cached in L2
       - Total: 4 * tokens * hidden_dim bytes per call

    4. act_and_mul_kernel (SiLU for SwiGLU):
       - Reads: gate (BF16) + up (BF16) = 2 * tokens * intermediate_size * 2 bytes
       - Writes: output (BF16) = tokens * intermediate_size * 2 bytes
       - Total: 6 * tokens * intermediate_size bytes per call

    Args:
        config: LlamaModelConfig with model architecture

    Returns:
        Dict with kernel data: bytes_per_call, total_calls, total_bytes
    """
    h = config.hidden_dim
    ff = config.intermediate_size
    t = config.tokens
    db = config.dtype_bytes

    # Bytes per single kernel call
    bytes_per_call = {
        # Quantize: read bf16 (2 bytes), write fp4 packed (0.5 bytes) + scales (~negligible)
        "quantize": int(t * h * 2.5),
        # Fused add + RMSNorm: read x + read residual + write output (all bf16)
        "fused_add_rms_norm": t * h * db * 3,
        # RMSNorm (standalone): read input + write output (both bf16)
        "rms_norm": t * h * db * 2,
        # SiLU (SwiGLU): read gate + up, write output (all bf16)
        "silu": t * ff * db * 3,
    }

    calls = count_llama_kernel_calls(config)

    return {
        kernel: {
            "bytes_per_call": bytes_per_call[kernel],
            "total_calls": calls[kernel],
            "total_bytes": bytes_per_call[kernel] * calls[kernel],
        }
        for kernel in bytes_per_call
    }


def calculate_kernel_bandwidth(
    name: str,
    time_s: float,
    total_bytes: int,
    peak_gbs: float,
    num_calls: int = 0,
    bytes_per_call: int = 0,
) -> KernelBandwidth:
    """Calculate bandwidth metrics for a kernel.

    Args:
        name: Kernel name
        time_s: Execution time in seconds
        total_bytes: Total bytes transferred
        peak_gbs: Theoretical peak bandwidth in GB/s
        num_calls: Number of kernel invocations
        bytes_per_call: Bytes per single invocation

    Returns:
        KernelBandwidth with calculated metrics
    """
    bandwidth_gbs = total_bytes / time_s / 1e9 if time_s > 0 else 0
    efficiency_pct = (bandwidth_gbs / peak_gbs * 100) if peak_gbs > 0 else 0

    return KernelBandwidth(
        name=name,
        time_s=time_s,
        total_bytes=total_bytes,
        bandwidth_gbs=bandwidth_gbs,
        efficiency_pct=efficiency_pct,
        num_calls=num_calls,
        bytes_per_call=bytes_per_call,
    )


def calculate_sglang_kernel_bandwidths(
    config: LlamaModelConfig,
    peak_gbs: float,
    quantize_time_s: float,
    layernorm_time_s: float,
    silu_time_s: float,
    fused_add_rms_norm_time_s: Optional[float] = None,
    quantize_calls: Optional[int] = None,
    layernorm_calls: Optional[int] = None,
    silu_calls: Optional[int] = None,
    fused_add_rms_norm_calls: Optional[int] = None,
) -> Tuple[List[KernelBandwidth], Dict[str, int]]:
    """Calculate bandwidth for SGLang element-wise kernels.

    Args:
        config: LlamaModelConfig with model architecture
        peak_gbs: Peak GPU bandwidth in GB/s
        quantize_time_s: Measured time for quantize kernels in seconds
        layernorm_time_s: Measured time for standalone RMSNorm kernels in seconds
        silu_time_s: Measured time for SiLU/act_and_mul kernels in seconds
        fused_add_rms_norm_time_s: Measured time for FusedAddRMSNormKernel (flashinfer)
        quantize_calls: Override estimated quantize calls (from nsys)
        layernorm_calls: Override estimated layernorm calls (from nsys)
        silu_calls: Override estimated SiLU calls (from nsys)
        fused_add_rms_norm_calls: Override estimated fused_add_rms_norm calls (from nsys)

    Returns:
        Tuple of (list of KernelBandwidth results, dict of kernel call counts)
    """
    # Calculate architecture-based byte estimates
    kernel_data = calculate_llama_kernel_bytes(config)
    estimated_calls = count_llama_kernel_calls(config)

    # Use nsys counts if provided, otherwise use estimated
    calls = {
        "quantize": quantize_calls if quantize_calls is not None else estimated_calls["quantize"],
        "fused_add_rms_norm": fused_add_rms_norm_calls if fused_add_rms_norm_calls is not None else estimated_calls["fused_add_rms_norm"],
        "rms_norm": layernorm_calls if layernorm_calls is not None else estimated_calls["rms_norm"],
        "silu": silu_calls if silu_calls is not None else estimated_calls["silu"],
    }

    # Map kernel names to their display names and timings
    # Only include kernels with non-zero time
    kernel_info = [
        ("quantize", "cvt_fp16_to_fp4 (quantize)", quantize_time_s),
    ]

    # Add fused kernel if time provided
    if fused_add_rms_norm_time_s is not None and fused_add_rms_norm_time_s > 0:
        kernel_info.append(("fused_add_rms_norm", "FusedAddRMSNormKernel (flashinfer)", fused_add_rms_norm_time_s))

    # Add standalone rms_norm if time provided and > 0
    if layernorm_time_s > 0:
        kernel_info.append(("rms_norm", "RMSNormKernel (flashinfer)", layernorm_time_s))

    kernel_info.append(("silu", "act_and_mul_kernel (flashinfer)", silu_time_s))

    results = []
    for key, display_name, time_s in kernel_info:
        # If using nsys call counts, recalculate total bytes
        bytes_per_call = kernel_data[key]["bytes_per_call"]
        total_bytes = bytes_per_call * calls[key]

        results.append(
            calculate_kernel_bandwidth(
                name=display_name,
                time_s=time_s,
                total_bytes=total_bytes,
                peak_gbs=peak_gbs,
                num_calls=calls[key],
                bytes_per_call=bytes_per_call,
            )
        )

    return results, calls


def generate_sglang_bandwidth_report(
    results: List[KernelBandwidth],
    calls: Dict[str, int],
    config: LlamaModelConfig,
    peak_gbs: float,
    calls_from_nsys: bool = False,
) -> str:
    """Generate formatted bandwidth report for SGLang kernel profiling.

    Args:
        results: List of KernelBandwidth results
        calls: Dict of kernel call counts
        config: LlamaModelConfig with model architecture
        peak_gbs: Theoretical peak bandwidth
        calls_from_nsys: Whether call counts came from nsys (vs estimated)

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 90)
    lines.append(f"SGLang Element-wise Kernel Bandwidth Report (B200 Peak: {peak_gbs:.0f} GB/s)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Model: Llama 3.1 8B")
    lines.append(
        f"Tokens: {config.tokens}, Hidden: {config.hidden_dim}, "
        f"Intermediate: {config.intermediate_size}, Layers: {config.num_layers}"
    )
    call_source = "nsys" if calls_from_nsys else "estimated"
    lines.append(f"Kernel call counts: {call_source}")
    lines.append("")
    lines.append(
        f"{'Kernel':<35s} {'Time (s)':>12s} {'Total Bytes':>14s} "
        f"{'Bandwidth':>12s} {'Efficiency':>10s}"
    )
    lines.append("-" * 90)

    total_time_s = 0
    total_bytes = 0
    weighted_bandwidth = 0

    for result in results:
        size_str = f"{result.total_bytes / 1e9:.4f} GB"
        bw_str = f"{result.bandwidth_gbs:.0f} GB/s"
        eff_str = f"{result.efficiency_pct:.1f}%"
        lines.append(
            f"{result.name:<35s} {result.time_s:>12.6f} {size_str:>14s} "
            f"{bw_str:>12s} {eff_str:>10s}"
        )
        total_time_s += result.time_s
        total_bytes += result.total_bytes
        weighted_bandwidth += result.bandwidth_gbs * result.time_s

    lines.append("-" * 90)

    # Summary
    total_size_str = f"{total_bytes / 1e9:.4f} GB"
    avg_bandwidth = weighted_bandwidth / total_time_s if total_time_s > 0 else 0
    avg_efficiency = (avg_bandwidth / peak_gbs * 100) if peak_gbs > 0 else 0

    lines.append(
        f"{'Total element-wise':<35s} {total_time_s:>12.6f} {total_size_str:>14s}"
    )
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Combined element-wise time: {total_time_s:.6f} s ({total_time_s * 1000:.3f} ms)")
    lines.append(f"  Total memory traffic: {total_bytes / 1e9:.4f} GB")
    lines.append(f"  Weighted average bandwidth: {avg_bandwidth:.0f} GB/s")
    lines.append(f"  Average efficiency: {avg_efficiency:.1f}% of {peak_gbs:.0f} GB/s theoretical")
    lines.append("")
    lines.append("Kernel Call Counts:")
    kernel_names = ["quantize", "fused_add_rms_norm", "rms_norm", "silu"]
    for kernel in kernel_names:
        if kernel in calls:
            lines.append(f"  {kernel}: {calls[kernel]} calls")
    lines.append("")

    # Per-call byte breakdown
    lines.append("Bytes per Call (estimated from architecture):")
    for result in results:
        lines.append(f"  {result.name}: {result.bytes_per_call / 1e6:.3f} MB")
    lines.append("")

    # Check for anomalies (efficiency > 100%)
    anomalies = [r for r in results if r.efficiency_pct > 100]
    if anomalies:
        lines.append("NOTE: Some kernels show >100% efficiency, which indicates:")
        lines.append("  - Timing may be inaccurate (kernel overlapped with other ops)")
        lines.append("  - Operations may be fused (fewer memory accesses than estimated)")
        lines.append("  - The kernel may use L2 cache effectively (less DRAM traffic)")
        lines.append("Anomalous kernels:")
        for r in anomalies:
            lines.append(f"  - {r.name}: {r.efficiency_pct:.1f}% efficiency")
        lines.append("")

    lines.append("=" * 90)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Memory Bandwidth Profiler for SGLang Element-wise Kernels (Llama 3.1 8B)"
    )

    # Model configuration
    parser.add_argument(
        "--tokens",
        type=int,
        default=4096,
        help="Number of input tokens (prefill sequence length, default: 4096)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Model hidden dimension (default: 4096 for Llama 8B)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=14336,
        help="FFN intermediate dimension (default: 14336 for Llama 8B)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of transformer layers (default: 32 for Llama 8B)",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=32,
        help="Number of attention heads (default: 32 for Llama 8B)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads for GQA (default: 8 for Llama 8B)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Dimension per attention head (default: 128)",
    )

    # GPU configuration
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=GPU_PEAK_BANDWIDTH["B200"],
        help="Peak GPU memory bandwidth in GB/s (default: 8000 for B200)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=list(GPU_PEAK_BANDWIDTH.keys()),
        help="Select GPU type for peak bandwidth (overrides --peak-bandwidth)",
    )

    # Kernel timings (required)
    parser.add_argument(
        "--quantize-time",
        type=float,
        default=0.002484,
        help="Time for quantize (cvt_fp16_to_fp4) kernels in seconds (default: 0.002484)",
    )
    parser.add_argument(
        "--layernorm-time",
        type=float,
        default=0.000020992,
        help="Time for RMSNorm kernels in seconds (default: 0.000020992)",
    )
    parser.add_argument(
        "--silu-time",
        type=float,
        default=0.004576,
        help="Time for SiLU/act_and_mul kernels in seconds (default: 0.004576)",
    )
    parser.add_argument(
        "--fused-add-rms-norm-time",
        type=float,
        default=None,
        help="Time for FusedAddRMSNormKernel (flashinfer fuses residual + norm)",
    )

    # Optional: override kernel call counts from nsys
    parser.add_argument(
        "--quantize-calls",
        type=int,
        default=None,
        help="Actual quantize kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--layernorm-calls",
        type=int,
        default=None,
        help="Actual RMSNorm kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--silu-calls",
        type=int,
        default=None,
        help="Actual SiLU kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--fused-add-rms-norm-calls",
        type=int,
        default=None,
        help="Call count for FusedAddRMSNormKernel (typically num_layers * 2)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for report (default: stdout)",
    )

    args = parser.parse_args()

    # Override peak bandwidth if GPU specified
    if args.gpu:
        args.peak_bandwidth = GPU_PEAK_BANDWIDTH[args.gpu]

    # Build model config
    config = LlamaModelConfig(
        hidden_dim=args.hidden_dim,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        tokens=args.tokens,
    )

    # Calculate bandwidths
    results, calls = calculate_sglang_kernel_bandwidths(
        config=config,
        peak_gbs=args.peak_bandwidth,
        quantize_time_s=args.quantize_time,
        layernorm_time_s=args.layernorm_time,
        silu_time_s=args.silu_time,
        fused_add_rms_norm_time_s=args.fused_add_rms_norm_time,
        quantize_calls=args.quantize_calls,
        layernorm_calls=args.layernorm_calls,
        silu_calls=args.silu_calls,
        fused_add_rms_norm_calls=args.fused_add_rms_norm_calls,
    )

    # Check if any nsys counts were provided
    calls_from_nsys = any(
        [
            args.quantize_calls is not None,
            args.layernorm_calls is not None,
            args.silu_calls is not None,
            args.fused_add_rms_norm_calls is not None,
        ]
    )

    # Generate report
    report = generate_sglang_bandwidth_report(
        results=results,
        calls=calls,
        config=config,
        peak_gbs=args.peak_bandwidth,
        calls_from_nsys=calls_from_nsys,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
