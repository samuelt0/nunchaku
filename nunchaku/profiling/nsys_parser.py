"""
Parser for Nsight Systems SQLite export files.

Extracts kernel timing information from nsys profile exports for bandwidth analysis.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Kernel name patterns for element-wise operations
ELEMENTWISE_KERNELS = {
    # Activation kernels
    "silu": {"pattern": "silu", "op_type": "activation"},
    "gelu": {"pattern": "gelu", "op_type": "activation"},
    "relu": {"pattern": "relu", "op_type": "activation"},
    "swiglu": {"pattern": "swiglu", "op_type": "activation"},
    # Normalization kernels
    "rmsnorm": {"pattern": "rms_norm", "op_type": "layernorm"},
    "layernorm": {"pattern": "layer_norm", "op_type": "layernorm"},
    "layernorm_general": {"pattern": "generallayernorm", "op_type": "layernorm"},
    "fused_norm": {"pattern": "fused_add_norm", "op_type": "layernorm"},
    "adaln": {"pattern": "adaln", "op_type": "layernorm"},
    "ada_layer_norm": {"pattern": "ada_layer_norm", "op_type": "layernorm"},
    # Element-wise operations
    "add": {"pattern": "elementwise_add", "op_type": "elementwise"},
    "mul": {"pattern": "elementwise_mul", "op_type": "elementwise"},
    "fma": {"pattern": "_fma_", "op_type": "elementwise"},
    "residual": {"pattern": "residual", "op_type": "elementwise"},
    "gate": {"pattern": "gate", "op_type": "elementwise"},
    # Memory operations
    "copy": {"pattern": "copy_kernel", "op_type": "memory"},
    "fill": {"pattern": "fill_kernel", "op_type": "memory"},
}

# Patterns that indicate GEMM/compute-bound kernels (not element-wise)
GEMM_PATTERNS = [
    "gemm",
    "gemv",
    "matmul",
    "cutlass",
    "cublas",
    "w4a4",
    "w8a8",
    "sm80_",
    "sm89_",
    "sm90_",
    "flash",
    "attention",
]


@dataclass
class KernelTiming:
    """Timing information for a single kernel execution.

    Attributes:
        name: Full kernel name from profiler
        short_name: Shortened name for display
        duration_ns: Duration in nanoseconds
        start_ns: Start timestamp in nanoseconds
        op_type: Type of operation (activation, layernorm, elementwise, gemm, other)
        is_elementwise: Whether this is an element-wise (memory-bound) kernel
        estimated_bytes: Estimated bytes read/written (if calculable)
    """

    name: str
    short_name: str
    duration_ns: float
    start_ns: float
    op_type: str
    is_elementwise: bool
    estimated_bytes: Optional[int] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns * 1e-6

    def __str__(self) -> str:
        return f"{self.short_name}: {self.duration_ms:.3f} ms ({self.op_type})"


def classify_kernel(name: str) -> Tuple[str, bool]:
    """Classify a kernel as element-wise or compute-bound.

    Args:
        name: Full kernel name

    Returns:
        Tuple of (op_type, is_elementwise)
    """
    name_lower = name.lower()

    # Check for GEMM/compute patterns first
    for pattern in GEMM_PATTERNS:
        if pattern in name_lower:
            return ("gemm", False)

    # Check for element-wise patterns
    for key, info in ELEMENTWISE_KERNELS.items():
        if info["pattern"].lower() in name_lower:
            return (info["op_type"], True)

    return ("other", False)


def shorten_kernel_name(name: str, max_len: int = 40) -> str:
    """Shorten kernel name for display.

    Args:
        name: Full kernel name
        max_len: Maximum length

    Returns:
        Shortened name
    """
    # Remove common prefixes/suffixes
    name = name.replace("void ", "")
    name = name.replace("__cuda_", "")

    # Remove template parameters for readability
    if "<" in name and ">" in name:
        # Keep just the base name and first template param
        base = name.split("<")[0]
        template_content = name[name.find("<") + 1 : name.rfind(">")]
        first_param = template_content.split(",")[0].strip()
        if len(first_param) > 20:
            first_param = first_param[:17] + "..."
        name = f"{base}<{first_param}>"

    if len(name) > max_len:
        name = name[: max_len - 3] + "..."

    return name


def parse_nsys_sqlite(db_path: str) -> List[KernelTiming]:
    """Extract kernel timings from nsys SQLite export.

    Args:
        db_path: Path to the .sqlite file exported by nsys

    Returns:
        List of KernelTiming objects sorted by start time
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    # Try different table names used by nsys versions
    kernel_table = None
    for table_name in [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        "NVTX_EVENTS",
    ]:
        if table_name in tables:
            kernel_table = table_name
            break

    if kernel_table is None:
        conn.close()
        raise ValueError(f"No kernel table found in database. Available tables: {tables}")

    # Query kernel information
    if kernel_table.startswith("CUPTI"):
        # Standard CUPTI kernel table
        query = """
            SELECT shortName, demangledName, end - start as duration, start
            FROM {}
            ORDER BY start
        """.format(kernel_table)

        # Try with shortName first, fall back to demangledName
        try:
            cursor.execute(query)
        except sqlite3.OperationalError:
            query = """
                SELECT demangledName, demangledName, end - start as duration, start
                FROM {}
                ORDER BY start
            """.format(kernel_table)
            cursor.execute(query)
    else:
        conn.close()
        raise ValueError(f"Unsupported table format: {kernel_table}")

    results = []
    for row in cursor.fetchall():
        short_name_raw, full_name, duration_ns, start_ns = row

        if full_name is None:
            full_name = short_name_raw or "unknown"
        if short_name_raw is None:
            short_name_raw = full_name

        op_type, is_elementwise = classify_kernel(full_name)

        results.append(
            KernelTiming(
                name=full_name,
                short_name=shorten_kernel_name(short_name_raw),
                duration_ns=float(duration_ns),
                start_ns=float(start_ns),
                op_type=op_type,
                is_elementwise=is_elementwise,
            )
        )

    conn.close()
    return results


def aggregate_kernel_timings(
    timings: List[KernelTiming],
) -> Dict[str, Dict[str, float]]:
    """Aggregate kernel timings by name.

    Args:
        timings: List of KernelTiming objects

    Returns:
        Dict mapping kernel short name to aggregated stats:
        - count: Number of invocations
        - total_ns: Total time in nanoseconds
        - avg_ns: Average time per invocation
        - min_ns: Minimum time
        - max_ns: Maximum time
    """
    aggregated: Dict[str, Dict[str, float]] = {}

    for timing in timings:
        key = timing.short_name
        if key not in aggregated:
            aggregated[key] = {
                "count": 0,
                "total_ns": 0,
                "min_ns": float("inf"),
                "max_ns": 0,
                "op_type": timing.op_type,
                "is_elementwise": timing.is_elementwise,
            }

        aggregated[key]["count"] += 1
        aggregated[key]["total_ns"] += timing.duration_ns
        aggregated[key]["min_ns"] = min(aggregated[key]["min_ns"], timing.duration_ns)
        aggregated[key]["max_ns"] = max(aggregated[key]["max_ns"], timing.duration_ns)

    # Calculate averages
    for key in aggregated:
        aggregated[key]["avg_ns"] = aggregated[key]["total_ns"] / aggregated[key]["count"]

    return aggregated


def filter_elementwise_kernels(timings: List[KernelTiming]) -> List[KernelTiming]:
    """Filter to only element-wise (memory-bound) kernels.

    Args:
        timings: List of all kernel timings

    Returns:
        Filtered list containing only element-wise kernels
    """
    return [t for t in timings if t.is_elementwise]


def summarize_by_op_type(
    timings: List[KernelTiming],
) -> Dict[str, Dict[str, float]]:
    """Summarize timings by operation type.

    Args:
        timings: List of kernel timings

    Returns:
        Dict mapping op_type to aggregated stats
    """
    summary: Dict[str, Dict[str, float]] = {}

    for timing in timings:
        op_type = timing.op_type
        if op_type not in summary:
            summary[op_type] = {"count": 0, "total_ns": 0}

        summary[op_type]["count"] += 1
        summary[op_type]["total_ns"] += timing.duration_ns

    # Calculate percentages
    total_ns = sum(s["total_ns"] for s in summary.values())
    for op_type in summary:
        summary[op_type]["total_ms"] = summary[op_type]["total_ns"] * 1e-6
        if total_ns > 0:
            summary[op_type]["pct"] = summary[op_type]["total_ns"] / total_ns * 100
        else:
            summary[op_type]["pct"] = 0

    return summary
