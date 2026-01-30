"""
Profiling utilities for Nunchaku models.

This module provides tools for capturing tensor shapes, calculating memory bandwidth,
and parsing profiling data from nsys/ncu.
"""

from .shape_capture import ShapeCapture, TensorShape
from .bandwidth_profiler import (
    calculate_bandwidth,
    calculate_bandwidth_from_ms,
    BandwidthResult,
    GPU_PEAK_BANDWIDTH,
    KernelBandwidth,
    calculate_quantize_kernel_bandwidth,
    calculate_layernorm_bandwidth,
    calculate_mul_add_bandwidth,
    calculate_add_kernel_bandwidth,
    count_flux_kernel_calls,
    calculate_kernel_bandwidths,
    calculate_bandwidth_from_captured_shapes,
    calculate_bandwidth_from_architecture,
    generate_kernel_bandwidth_report,
)
from .kernel_shape_capture import (
    KernelCall,
    KernelShapeCapture,
    capture_kernel_shapes,
    FluxModelConfig,
    calculate_flux_kernel_bytes,
)
from .nsys_parser import parse_nsys_sqlite, KernelTiming

__all__ = [
    "ShapeCapture",
    "TensorShape",
    "calculate_bandwidth",
    "calculate_bandwidth_from_ms",
    "BandwidthResult",
    "GPU_PEAK_BANDWIDTH",
    "parse_nsys_sqlite",
    "KernelTiming",
    "KernelBandwidth",
    "calculate_quantize_kernel_bandwidth",
    "calculate_layernorm_bandwidth",
    "calculate_mul_add_bandwidth",
    "calculate_add_kernel_bandwidth",
    "count_flux_kernel_calls",
    "calculate_kernel_bandwidths",
    "calculate_bandwidth_from_captured_shapes",
    "calculate_bandwidth_from_architecture",
    "generate_kernel_bandwidth_report",
    "KernelCall",
    "KernelShapeCapture",
    "capture_kernel_shapes",
    "FluxModelConfig",
    "calculate_flux_kernel_bytes",
]
