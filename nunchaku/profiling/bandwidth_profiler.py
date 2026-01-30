"""
Memory bandwidth calculation utilities.

Provides functions to calculate memory bandwidth utilization and compare
against theoretical peak bandwidth.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .nsys_parser import KernelTiming


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
class BandwidthResult:
    """Results from bandwidth calculation.

    Attributes:
        bytes_total: Total bytes transferred
        time_ns: Time in nanoseconds
        time_ms: Time in milliseconds
        bandwidth_gbs: Achieved bandwidth in GB/s
        peak_gbs: Theoretical peak bandwidth in GB/s
        efficiency_pct: Bandwidth efficiency as percentage of peak
    """

    bytes_total: int
    time_ns: float
    time_ms: float
    bandwidth_gbs: float
    peak_gbs: float
    efficiency_pct: float

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_mb = self.bytes_total / (1024 * 1024)
        return (
            f"{self.time_ms:.3f} ms | {size_mb:.1f} MB | "
            f"{self.bandwidth_gbs:.0f} GB/s | {self.efficiency_pct:.1f}% eff"
        )


def calculate_bandwidth(
    bytes_total: int,
    time_ns: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> BandwidthResult:
    """Calculate memory bandwidth from bytes and time.

    Args:
        bytes_total: Total bytes transferred (read + write)
        time_ns: Execution time in nanoseconds
        peak_gbs: Theoretical peak bandwidth in GB/s (default: B200's 8000 GB/s)

    Returns:
        BandwidthResult with calculated metrics
    """
    time_s = time_ns * 1e-9
    time_ms = time_ns * 1e-6
    bandwidth_gbs = bytes_total / time_s / 1e9 if time_s > 0 else 0
    efficiency_pct = (bandwidth_gbs / peak_gbs * 100) if peak_gbs > 0 else 0

    return BandwidthResult(
        bytes_total=bytes_total,
        time_ns=time_ns,
        time_ms=time_ms,
        bandwidth_gbs=bandwidth_gbs,
        peak_gbs=peak_gbs,
        efficiency_pct=efficiency_pct,
    )


def calculate_bandwidth_from_ms(
    bytes_total: int,
    time_ms: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> BandwidthResult:
    """Calculate memory bandwidth from bytes and time in milliseconds.

    Args:
        bytes_total: Total bytes transferred (read + write)
        time_ms: Execution time in milliseconds
        peak_gbs: Theoretical peak bandwidth in GB/s

    Returns:
        BandwidthResult with calculated metrics
    """
    return calculate_bandwidth(bytes_total, time_ms * 1e6, peak_gbs)


@dataclass
class LayerBandwidth:
    """Bandwidth results for a transformer layer.

    Attributes:
        layer_idx: Index of the layer
        layer_type: "double" or "single"
        time_ms: Execution time in milliseconds
        input_shape: Shape of input hidden states
        input_bytes: Bytes in input tensor
        estimated_elementwise_bytes: Estimated bytes for element-wise ops
        bandwidth_result: BandwidthResult for this layer (if calculated)
    """

    layer_idx: int
    layer_type: str
    time_ms: float
    input_shape: tuple
    input_bytes: int
    estimated_elementwise_bytes: Optional[int] = None
    bandwidth_result: Optional[BandwidthResult] = None

    def __str__(self) -> str:
        dtype_str = "bf16"
        if self.bandwidth_result:
            return (
                f"Layer {self.layer_idx:2d} ({self.layer_type:6s}): "
                f"Time: {self.time_ms:.2f} ms | "
                f"Input: {self.input_shape} {dtype_str} | "
                f"BW: ~{self.bandwidth_result.bandwidth_gbs:.0f} GB/s"
            )
        return (
            f"Layer {self.layer_idx:2d} ({self.layer_type:6s}): "
            f"Time: {self.time_ms:.2f} ms | "
            f"Input: {self.input_shape} {dtype_str}"
        )


def estimate_layer_elementwise_bytes(
    hidden_shape: tuple,
    encoder_shape: tuple,
    layer_type: str = "double",
    dtype_bytes: int = 2,
) -> int:
    """Estimate total element-wise bytes for a transformer layer.

    For a double block, estimates:
    - 2x LayerNorm (img + txt)
    - 2x gate multiply
    - 2x residual add
    - concat/split operations
    - MLP activations

    For a single block, estimates:
    - 1x LayerNorm
    - 1x gate multiply
    - 1x residual add
    - MLP activations

    Args:
        hidden_shape: Shape of hidden_states (batch, img_tokens, hidden_dim)
        encoder_shape: Shape of encoder_hidden_states (batch, txt_tokens, hidden_dim)
        layer_type: "double" or "single"
        dtype_bytes: Bytes per element (2 for bf16/fp16)

    Returns:
        Estimated total bytes for element-wise operations
    """
    batch, img_tokens, hidden_dim = hidden_shape
    _, txt_tokens, _ = encoder_shape

    img_size = batch * img_tokens * hidden_dim * dtype_bytes
    txt_size = batch * txt_tokens * hidden_dim * dtype_bytes

    if layer_type == "double":
        # Double block operations (approximate):
        # - 2x AdaLayerNorm (img norm, txt norm): ~4 * (img + txt) bytes
        # - Gate multiply for img and txt: ~3 * (img + txt)
        # - Residual adds: ~3 * (img + txt)
        # - Concat before attention: ~2 * (img + txt)
        # - Split after attention: ~2 * (img + txt)
        # - MLP activations (SiLU, mul): ~3 * img for FF
        # Total rough estimate: ~17x combined size
        total = 17 * (img_size + txt_size)
    else:
        # Single block (operates on concatenated img+txt):
        combined_size = batch * (img_tokens + txt_tokens) * hidden_dim * dtype_bytes
        # - 1x LayerNorm: ~2 * combined
        # - Gate multiply: ~3 * combined
        # - Residual add: ~3 * combined
        # - MLP: ~3 * combined
        total = 11 * combined_size

    return total


@dataclass
class KernelBandwidth:
    """Bandwidth results for a specific kernel type.

    Attributes:
        name: Kernel name (e.g., "quantize_w4a4", "generalLayerNorm")
        time_s: Total execution time in seconds
        total_bytes: Total bytes transferred (read + write)
        bandwidth_gbs: Achieved bandwidth in GB/s
        efficiency_pct: Bandwidth efficiency as percentage of peak
        num_calls: Number of kernel invocations
    """

    name: str
    time_s: float
    total_bytes: int
    bandwidth_gbs: float
    efficiency_pct: float
    num_calls: int = 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_gb = self.total_bytes / 1e9
        return (
            f"{self.name:30s} {self.time_s:.4f} s | {size_gb:.2f} GB | "
            f"{self.bandwidth_gbs:.0f} GB/s | {self.efficiency_pct:.1f}%"
        )


def _calc_kernel_bandwidth(
    name: str,
    time_s: float,
    total_bytes: int,
    peak_gbs: float,
    num_calls: int = 0,
) -> KernelBandwidth:
    """Helper to calculate kernel bandwidth metrics.

    Args:
        name: Kernel name
        time_s: Execution time in seconds
        total_bytes: Total bytes transferred
        peak_gbs: Theoretical peak bandwidth in GB/s
        num_calls: Number of kernel invocations

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
    )


def calculate_quantize_kernel_bandwidth(
    M: int,
    K: int,
    num_calls: int,
    time_s: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> KernelBandwidth:
    """Calculate bandwidth for quantize_w4a4_fuse_lora_kernel.

    Memory pattern:
    - Reads: input (BF16) = M × K × 2 bytes
    - Writes: output (INT4 packed) = M × K / 2 bytes, plus scales (small)

    Args:
        M: First dimension (tokens)
        K: Second dimension (hidden_dim or projection dim)
        num_calls: Number of kernel invocations
        time_s: Total execution time in seconds
        peak_gbs: Theoretical peak bandwidth in GB/s

    Returns:
        KernelBandwidth with calculated metrics
    """
    # 2 bytes read (BF16) + 0.5 bytes write (INT4 packed) ≈ 2.5 bytes per element
    bytes_per_call = int(M * K * 2.5)
    total_bytes = bytes_per_call * num_calls
    return _calc_kernel_bandwidth(
        "quantize_w4a4_fuse_lora", time_s, total_bytes, peak_gbs, num_calls
    )


def calculate_layernorm_bandwidth(
    tokens: int,
    hidden_dim: int,
    num_calls: int,
    time_s: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> KernelBandwidth:
    """Calculate bandwidth for generalLayerNorm kernel.

    Memory pattern:
    - Reads: input (BF16) = tokens × hidden_dim × 2 bytes
    - Writes: output (BF16) = tokens × hidden_dim × 2 bytes
    - gamma/beta are small and typically cached in L2

    Args:
        tokens: Number of tokens (batch * sequence_length)
        hidden_dim: Hidden dimension size
        num_calls: Number of kernel invocations
        time_s: Total execution time in seconds
        peak_gbs: Theoretical peak bandwidth in GB/s

    Returns:
        KernelBandwidth with calculated metrics
    """
    # Read input + write output = 4 bytes per element
    bytes_per_call = tokens * hidden_dim * 4
    total_bytes = bytes_per_call * num_calls
    return _calc_kernel_bandwidth(
        "generalLayerNorm", time_s, total_bytes, peak_gbs, num_calls
    )


def calculate_mul_add_bandwidth(
    batch: int,
    tokens: int,
    hidden_dim: int,
    num_calls: int,
    time_s: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> KernelBandwidth:
    """Calculate bandwidth for mul_add_kernel.

    Operation: x = x * (scale + 1) + bias (in-place with broadcast)

    Memory pattern:
    - Reads: x (full tensor), scale, bias (broadcasted from [batch, hidden_dim])
    - Writes: x (in-place)

    Args:
        batch: Batch size
        tokens: Number of tokens per batch
        hidden_dim: Hidden dimension size
        num_calls: Number of kernel invocations
        time_s: Total execution time in seconds
        peak_gbs: Theoretical peak bandwidth in GB/s

    Returns:
        KernelBandwidth with calculated metrics
    """
    # x is [batch, tokens, hidden_dim], scale/bias are [batch, hidden_dim]
    # Read x + write x (in-place)
    x_bytes = batch * tokens * hidden_dim * 2 * 2  # read + write
    # scale + bias (small, likely cached, but counted conservatively)
    scale_bias_bytes = batch * hidden_dim * 2 * 2
    bytes_per_call = x_bytes + scale_bias_bytes
    total_bytes = bytes_per_call * num_calls
    return _calc_kernel_bandwidth(
        "mul_add_kernel", time_s, total_bytes, peak_gbs, num_calls
    )


def calculate_add_kernel_bandwidth(
    length: int,
    num_calls: int,
    time_s: float,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> KernelBandwidth:
    """Calculate bandwidth for add_kernel.

    Operation: c = a + b

    Memory pattern:
    - Reads: a, b
    - Writes: c

    Args:
        length: Total number of elements (batch * tokens * hidden_dim)
        num_calls: Number of kernel invocations
        time_s: Total execution time in seconds
        peak_gbs: Theoretical peak bandwidth in GB/s

    Returns:
        KernelBandwidth with calculated metrics
    """
    # Read a + b, write c = 3 × length × 2 bytes
    bytes_per_call = length * 2 * 3
    total_bytes = bytes_per_call * num_calls
    return _calc_kernel_bandwidth(
        "add_kernel", time_s, total_bytes, peak_gbs, num_calls
    )


def count_flux_kernel_calls(
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
    num_steps: int = 50,
) -> Dict[str, int]:
    """Estimate kernel calls across a full Flux diffusion run.

    NOTE: These are estimates. For accurate counts, use nsys profiling data
    and pass the actual counts to calculate_kernel_bandwidths().

    Based on FluxTransformerBlock and FluxSingleTransformerBlock structure.

    Args:
        num_double_blocks: Number of double transformer blocks (default: 19)
        num_single_blocks: Number of single transformer blocks (default: 38)
        num_steps: Number of diffusion steps (default: 50)

    Returns:
        Dict with kernel call counts: quantize, layernorm, mul_add, add
    """
    calls = {
        "quantize": 0,
        "layernorm": 0,
        "mul_add": 0,
        "add": 0,
    }

    # Double blocks
    for _ in range(num_double_blocks):
        calls["layernorm"] += 4  # norm1, norm1_context, norm2, norm2_context
        calls["quantize"] += 8  # QKV + out for img and txt, plus FF layers
        calls["mul_add"] += 6  # gate operations (msa_gate, mlp_gate, scale+shift)
        calls["add"] += 4  # residual connections

    # Single blocks
    for _ in range(num_single_blocks):
        calls["layernorm"] += 1  # norm
        calls["quantize"] += 4  # QKV + MLP projections
        calls["mul_add"] += 2  # gate operations
        calls["add"] += 1  # residual connection

    # Final norm
    calls["layernorm"] += 1

    # Multiply by number of diffusion steps
    for key in calls:
        calls[key] *= num_steps

    return calls


def calculate_kernel_bandwidths(
    img_tokens: int = 4096,
    txt_tokens: int = 512,
    hidden_dim: int = 3072,
    mlp_hidden_dim: int = 12288,  # FFN hidden dim, typically 4x hidden_dim
    batch: int = 1,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
    num_steps: int = 50,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    quantize_time_s: float = 0.25969,
    layernorm_time_s: float = 0.108147,
    mul_add_time_s: float = 0.105632,
    add_time_s: float = 0.099116,
    # Optional: override with actual nsys counts
    quantize_calls: Optional[int] = None,
    layernorm_calls: Optional[int] = None,
    mul_add_calls: Optional[int] = None,
    add_calls: Optional[int] = None,
) -> Tuple[List[KernelBandwidth], Dict[str, int]]:
    """Calculate bandwidth for each kernel type based on measured timings.

    Args:
        img_tokens: Number of image tokens (default: 4096 for 1024×1024)
        txt_tokens: Number of text tokens (default: 512)
        hidden_dim: Hidden dimension size (default: 3072)
        mlp_hidden_dim: FFN hidden dimension (default: 12288, typically 4x hidden_dim)
        batch: Batch size (default: 1)
        num_double_blocks: Number of double transformer blocks (default: 19)
        num_single_blocks: Number of single transformer blocks (default: 38)
        num_steps: Number of diffusion steps (default: 50)
        peak_gbs: Peak GPU bandwidth in GB/s (default: B200's 8000 GB/s)
        quantize_time_s: Measured time for quantize kernels in seconds
        layernorm_time_s: Measured time for layernorm kernels in seconds
        mul_add_time_s: Measured time for mul_add kernels in seconds
        add_time_s: Measured time for add kernels in seconds
        quantize_calls: Actual quantize kernel calls from nsys (overrides estimate)
        layernorm_calls: Actual layernorm kernel calls from nsys (overrides estimate)
        mul_add_calls: Actual mul_add kernel calls from nsys (overrides estimate)
        add_calls: Actual add kernel calls from nsys (overrides estimate)

    Returns:
        Tuple of (list of KernelBandwidth results, dict of kernel call counts)
    """
    # Use actual nsys counts if provided, otherwise estimate
    estimated_calls = count_flux_kernel_calls(num_double_blocks, num_single_blocks, num_steps)
    calls = {
        "quantize": quantize_calls if quantize_calls is not None else estimated_calls["quantize"],
        "layernorm": layernorm_calls if layernorm_calls is not None else estimated_calls["layernorm"],
        "mul_add": mul_add_calls if mul_add_calls is not None else estimated_calls["mul_add"],
        "add": add_calls if add_calls is not None else estimated_calls["add"],
    }

    combined_tokens = img_tokens + txt_tokens
    results = []

    # Quantize kernel (operates on activation matrices before GEMM)
    # Different projections have different K dimensions:
    # - Attention QKV: K = hidden_dim (3072) -> out = hidden_dim*3 (9216)
    # - Attention out: K = hidden_dim (3072)
    # - FFN up: K = hidden_dim (3072) -> out = mlp_hidden_dim (12288)
    # - FFN down: K = mlp_hidden_dim (12288) -> out = hidden_dim (3072)
    # Average K across projections (weighted by frequency)
    avg_K = (hidden_dim + mlp_hidden_dim) // 2  # rough average
    results.append(
        calculate_quantize_kernel_bandwidth(
            combined_tokens, avg_K, calls["quantize"], quantize_time_s, peak_gbs
        )
    )

    # LayerNorm - operates on [tokens, hidden_dim]
    results.append(
        calculate_layernorm_bandwidth(
            combined_tokens, hidden_dim, calls["layernorm"], layernorm_time_s, peak_gbs
        )
    )

    # mul_add (gate operations) - operates on [batch, tokens, hidden_dim]
    results.append(
        calculate_mul_add_bandwidth(
            batch, combined_tokens, hidden_dim, calls["mul_add"], mul_add_time_s, peak_gbs
        )
    )

    # add kernel (residual connections)
    length = batch * combined_tokens * hidden_dim
    results.append(
        calculate_add_kernel_bandwidth(length, calls["add"], add_time_s, peak_gbs)
    )

    return results, calls


def generate_kernel_bandwidth_report(
    results: List[KernelBandwidth],
    calls: Dict[str, int],
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    img_tokens: int = 4096,
    txt_tokens: int = 512,
    hidden_dim: int = 3072,
    mlp_hidden_dim: int = 12288,
    num_steps: int = 50,
    calls_from_nsys: bool = False,
) -> str:
    """Generate formatted bandwidth report for kernel profiling.

    Args:
        results: List of KernelBandwidth results
        calls: Dict of kernel call counts
        peak_gbs: Theoretical peak bandwidth
        img_tokens: Number of image tokens
        txt_tokens: Number of text tokens
        hidden_dim: Hidden dimension size
        mlp_hidden_dim: FFN hidden dimension
        num_steps: Number of diffusion steps
        calls_from_nsys: Whether call counts came from nsys (vs estimated)

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Element-wise Kernel Bandwidth Report (B200 Peak: {peak_gbs:.0f} GB/s)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  Image tokens: {img_tokens}, Text tokens: {txt_tokens}, Combined: {img_tokens + txt_tokens}")
    lines.append(f"  Hidden dim: {hidden_dim}, MLP hidden dim: {mlp_hidden_dim}")
    lines.append(f"  Diffusion steps: {num_steps}")
    call_source = "nsys" if calls_from_nsys else "estimated"
    lines.append(f"  Kernel call counts: {call_source}")
    lines.append("")
    lines.append(
        f"{'Kernel':<30s} {'Time (s)':>10s} {'Total Bytes':>14s} "
        f"{'Bandwidth':>12s} {'Efficiency':>10s}"
    )
    lines.append("-" * 80)

    total_time_s = 0
    total_bytes = 0
    weighted_bandwidth = 0

    for result in results:
        size_str = f"{result.total_bytes / 1e9:.2f} GB"
        bw_str = f"{result.bandwidth_gbs:.0f} GB/s"
        eff_str = f"{result.efficiency_pct:.1f}%"
        lines.append(
            f"{result.name:<30s} {result.time_s:>10.4f} {size_str:>14s} "
            f"{bw_str:>12s} {eff_str:>10s}"
        )
        total_time_s += result.time_s
        total_bytes += result.total_bytes
        weighted_bandwidth += result.bandwidth_gbs * result.time_s

    lines.append("-" * 80)

    # Summary
    total_size_str = f"{total_bytes / 1e9:.2f} GB"
    avg_bandwidth = weighted_bandwidth / total_time_s if total_time_s > 0 else 0
    avg_efficiency = (avg_bandwidth / peak_gbs * 100) if peak_gbs > 0 else 0

    lines.append(
        f"{'Total element-wise':<30s} {total_time_s:>10.4f} {total_size_str:>14s}"
    )
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Combined element-wise time: {total_time_s:.4f} s")
    lines.append(f"  Total memory traffic: {total_bytes / 1e9:.2f} GB")
    lines.append(f"  Weighted average bandwidth: {avg_bandwidth:.0f} GB/s")
    lines.append(f"  Average efficiency: {avg_efficiency:.1f}% of {peak_gbs:.0f} GB/s theoretical")
    lines.append("")
    lines.append("Kernel Call Counts:")
    for kernel, count in calls.items():
        lines.append(f"  {kernel}: {count} calls")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def calculate_bandwidth_from_captured_shapes(
    captured_bytes: Dict[str, int],
    captured_counts: Dict[str, int],
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    quantize_time_s: float = 0.25969,
    layernorm_time_s: float = 0.108147,
    mul_add_time_s: float = 0.105632,
    add_time_s: float = 0.099116,
    num_steps: int = 50,
) -> Tuple[List[KernelBandwidth], Dict[str, int]]:
    """Calculate bandwidth using actual captured byte counts from shape tracing.

    This function uses bytes calculated from actual tensor shapes observed during
    inference, combined with timing data from nsys profiling.

    Args:
        captured_bytes: Dict of kernel_name -> total bytes from one forward pass
        captured_counts: Dict of kernel_name -> call count from one forward pass
        peak_gbs: Peak GPU bandwidth in GB/s
        quantize_time_s: Measured time for quantize kernels in seconds
        layernorm_time_s: Measured time for layernorm kernels in seconds
        mul_add_time_s: Measured time for mul_add kernels in seconds
        add_time_s: Measured time for add kernels in seconds
        num_steps: Number of diffusion steps (multiplier for captured data)

    Returns:
        Tuple of (list of KernelBandwidth results, dict of kernel call counts)
    """
    # Map captured kernel names to timing parameters
    kernel_times = {
        "quantize_w4a4_fuse_lora": quantize_time_s,
        "generalLayerNorm": layernorm_time_s,
        "mul_add_kernel": mul_add_time_s,
        "add_kernel": add_time_s,
    }

    results = []
    calls = {}

    for kernel_name, time_s in kernel_times.items():
        if kernel_name in captured_bytes:
            # Scale by number of steps (captured data is for one step)
            total_bytes = captured_bytes[kernel_name] * num_steps
            num_calls = captured_counts.get(kernel_name, 0) * num_steps
            calls[kernel_name] = num_calls

            bandwidth_gbs = total_bytes / time_s / 1e9 if time_s > 0 else 0
            efficiency_pct = (bandwidth_gbs / peak_gbs * 100) if peak_gbs > 0 else 0

            results.append(KernelBandwidth(
                name=kernel_name,
                time_s=time_s,
                total_bytes=total_bytes,
                bandwidth_gbs=bandwidth_gbs,
                efficiency_pct=efficiency_pct,
                num_calls=num_calls,
            ))
        else:
            calls[kernel_name] = 0

    return results, calls


def calculate_bandwidth_from_architecture(
    img_tokens: int = 4096,
    txt_tokens: int = 512,
    hidden_dim: int = 3072,
    mlp_hidden_dim: int = 12288,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
    num_steps: int = 50,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
    quantize_time_s: float = 0.25969,
    layernorm_time_s: float = 0.108147,
    mul_add_time_s: float = 0.105632,
    add_time_s: float = 0.099116,
) -> Tuple[List[KernelBandwidth], Dict[str, int]]:
    """Calculate bandwidth based on Flux model architecture.

    This function computes exact memory traffic for each kernel type by analyzing
    the Flux transformer architecture and calculating bytes for each operation.

    Args:
        img_tokens: Number of image tokens (height/16 * width/16)
        txt_tokens: Number of text tokens (typically 512)
        hidden_dim: Model hidden dimension (3072 for Flux)
        mlp_hidden_dim: FFN hidden dimension (typically 4x hidden_dim)
        num_double_blocks: Number of double transformer blocks (19 for Flux)
        num_single_blocks: Number of single transformer blocks (38 for Flux)
        num_steps: Number of diffusion steps
        peak_gbs: Peak GPU bandwidth in GB/s
        quantize_time_s: Measured time for quantize kernels in seconds
        layernorm_time_s: Measured time for layernorm kernels in seconds
        mul_add_time_s: Measured time for mul_add kernels in seconds
        add_time_s: Measured time for add kernels in seconds

    Returns:
        Tuple of (list of KernelBandwidth results, dict of kernel call counts)
    """
    from .kernel_shape_capture import FluxModelConfig, calculate_flux_kernel_bytes

    config = FluxModelConfig(
        hidden_dim=hidden_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        num_double_blocks=num_double_blocks,
        num_single_blocks=num_single_blocks,
        img_tokens=img_tokens,
        txt_tokens=txt_tokens,
    )

    kernel_data = calculate_flux_kernel_bytes(config, num_steps)

    # Map kernel names to timing parameters
    kernel_times = {
        "quantize_w4a4_fuse_lora": quantize_time_s,
        "generalLayerNorm": layernorm_time_s,
        "mul_add_kernel": mul_add_time_s,
        "add_kernel": add_time_s,
    }

    results = []
    calls = {}

    for kernel_name, time_s in kernel_times.items():
        data = kernel_data.get(kernel_name, {})
        total_bytes = data.get("total_bytes", 0)
        num_calls = data.get("total_calls", 0)
        calls[kernel_name] = num_calls

        bandwidth_gbs = total_bytes / time_s / 1e9 if time_s > 0 else 0
        efficiency_pct = (bandwidth_gbs / peak_gbs * 100) if peak_gbs > 0 else 0

        results.append(KernelBandwidth(
            name=kernel_name,
            time_s=time_s,
            total_bytes=total_bytes,
            bandwidth_gbs=bandwidth_gbs,
            efficiency_pct=efficiency_pct,
            num_calls=num_calls,
        ))

    return results, calls


def generate_bandwidth_report(
    layer_results: List[LayerBandwidth],
    kernel_results: Optional[List["KernelTiming"]] = None,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> str:
    """Generate formatted bandwidth profiling report.

    Args:
        layer_results: List of LayerBandwidth results
        kernel_results: Optional list of KernelTiming results from nsys
        peak_gbs: Theoretical peak bandwidth for the GPU

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Memory Bandwidth Profiling Report (Peak: {peak_gbs:.0f} GB/s)")
    lines.append("=" * 80)
    lines.append("")

    # Per-layer summary
    lines.append("Per-Layer Summary (transformer forward):")
    total_time_ms = 0
    for result in layer_results:
        lines.append(f"  {result}")
        total_time_ms += result.time_ms
    lines.append("")
    lines.append(f"Total transformer time: {total_time_ms:.2f} ms")
    lines.append("")

    # Kernel breakdown if available
    if kernel_results:
        lines.append("Element-wise Kernel Breakdown (from nsys):")
        total_elementwise_ms = 0
        total_elementwise_bytes = 0
        elementwise_efficiencies = []

        for kernel in kernel_results:
            if kernel.is_elementwise:
                bw_result = calculate_bandwidth(
                    kernel.estimated_bytes or 0,
                    kernel.duration_ns,
                    peak_gbs,
                )
                lines.append(
                    f"  {kernel.short_name:30s}: {bw_result}"
                )
                total_elementwise_ms += bw_result.time_ms
                total_elementwise_bytes += kernel.estimated_bytes or 0
                elementwise_efficiencies.append(bw_result.efficiency_pct)

        lines.append("")
        if total_time_ms > 0:
            pct_of_total = total_elementwise_ms / total_time_ms * 100
            lines.append(
                f"Total element-wise time: {total_elementwise_ms:.2f} ms "
                f"(of {total_time_ms:.2f} ms total, {pct_of_total:.1f}%)"
            )
        if elementwise_efficiencies:
            avg_eff = sum(elementwise_efficiencies) / len(elementwise_efficiencies)
            lines.append(f"Avg element-wise efficiency: {avg_eff:.1f}%")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
