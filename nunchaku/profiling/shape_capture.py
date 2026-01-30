"""
Tensor shape capture utilities for profiling.

Captures tensor shapes at key boundaries during forward pass to enable
bandwidth calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class TensorShape:
    """Captures metadata about a tensor for bandwidth calculations.

    Attributes:
        name: Identifier for this tensor (e.g., "layer_0_input")
        shape: Tuple of tensor dimensions
        dtype: String representation of tensor dtype (e.g., "torch.bfloat16")
        bytes: Total size in bytes (numel * element_size)
    """

    name: str
    shape: tuple
    dtype: str
    bytes: int

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor) -> "TensorShape":
        """Create TensorShape from a torch tensor.

        Args:
            name: Identifier for this tensor
            tensor: The torch tensor to capture shape from

        Returns:
            TensorShape instance with captured metadata
        """
        return cls(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            bytes=tensor.numel() * tensor.element_size(),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_mb = self.bytes / (1024 * 1024)
        return f"{self.name}: {self.shape} {self.dtype} ({size_mb:.2f} MB)"


@dataclass
class ShapeCapture:
    """Captures tensor shapes during forward pass.

    Use this class to record tensor shapes at various points during model
    inference for later bandwidth analysis.

    Example:
        >>> capture = ShapeCapture()
        >>> capture.record("hidden_states", hidden_states)
        >>> capture.record("encoder_hidden", encoder_hidden_states)
        >>> print(capture.summary())
    """

    shapes: Dict[str, List[TensorShape]] = field(default_factory=dict)

    def record(self, name: str, tensor: torch.Tensor) -> TensorShape:
        """Record tensor shape with given name.

        Args:
            name: Identifier for this capture point
            tensor: The tensor to capture

        Returns:
            The captured TensorShape
        """
        shape = TensorShape.from_tensor(name, tensor)
        self.shapes.setdefault(name, []).append(shape)
        return shape

    def get(self, name: str) -> Optional[List[TensorShape]]:
        """Get all captures for a given name.

        Args:
            name: The identifier to look up

        Returns:
            List of TensorShape or None if not found
        """
        return self.shapes.get(name)

    def get_latest(self, name: str) -> Optional[TensorShape]:
        """Get the most recent capture for a given name.

        Args:
            name: The identifier to look up

        Returns:
            Most recent TensorShape or None if not found
        """
        captures = self.shapes.get(name)
        if captures:
            return captures[-1]
        return None

    def clear(self):
        """Clear all captured shapes."""
        self.shapes.clear()

    def total_bytes(self, names: Optional[List[str]] = None) -> int:
        """Calculate total bytes across captures.

        Args:
            names: Optional list of names to include. If None, includes all.

        Returns:
            Total bytes across all specified captures
        """
        total = 0
        for name, captures in self.shapes.items():
            if names is None or name in names:
                for shape in captures:
                    total += shape.bytes
        return total

    def summary(self) -> str:
        """Generate human-readable summary of all captures.

        Returns:
            Formatted string summary
        """
        lines = ["Captured Tensor Shapes:"]
        lines.append("-" * 60)
        total_bytes = 0
        for name, captures in self.shapes.items():
            for shape in captures:
                lines.append(str(shape))
                total_bytes += shape.bytes
        lines.append("-" * 60)
        lines.append(f"Total: {total_bytes / (1024 * 1024):.2f} MB")
        return "\n".join(lines)


def estimate_elementwise_bytes(
    hidden_shape: tuple,
    encoder_shape: tuple,
    temb_shape: tuple,
    dtype_bytes: int = 2,
) -> Dict[str, int]:
    """Estimate bytes read/written for common element-wise operations.

    For Flux transformer layers, estimates memory traffic for:
    - LayerNorm/RMSNorm
    - SiLU activation
    - SiLU+Mul fused
    - Residual add
    - Gate multiply

    Args:
        hidden_shape: Shape of hidden_states (batch, seq_len, hidden_dim)
        encoder_shape: Shape of encoder_hidden_states (batch, txt_tokens, hidden_dim)
        temb_shape: Shape of temporal embedding (batch, hidden_dim)
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32)

    Returns:
        Dict mapping operation name to estimated bytes
    """
    batch, img_tokens, hidden_dim = hidden_shape
    _, txt_tokens, _ = encoder_shape

    estimates = {}

    # LayerNorm/RMSNorm: read input, write output (same size)
    img_size = batch * img_tokens * hidden_dim * dtype_bytes
    txt_size = batch * txt_tokens * hidden_dim * dtype_bytes
    estimates["layernorm_img"] = 2 * img_size  # read + write
    estimates["layernorm_txt"] = 2 * txt_size

    # SiLU activation: read input, write output
    estimates["silu"] = 2 * img_size

    # SiLU+Mul fused: read 2 inputs, write 1 output
    estimates["silu_mul_fused"] = 3 * img_size

    # Residual add: read 2 inputs, write 1 output
    estimates["residual_add_img"] = 3 * img_size
    estimates["residual_add_txt"] = 3 * txt_size

    # Gate multiply: read input + gate (gate is broadcasted from temb)
    gate_size = batch * hidden_dim * dtype_bytes
    estimates["gate_multiply_img"] = 2 * img_size + gate_size
    estimates["gate_multiply_txt"] = 2 * txt_size + gate_size

    # Combined image tokens (cat then split)
    combined_size = batch * (img_tokens + txt_tokens) * hidden_dim * dtype_bytes
    estimates["concat"] = (img_size + txt_size) + combined_size  # read both, write combined
    estimates["split"] = combined_size + (img_size + txt_size)  # read combined, write both

    return estimates


def get_flux_shapes_for_resolution(height: int, width: int, batch: int = 1) -> Dict[str, tuple]:
    """Get expected tensor shapes for a given Flux image resolution.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        batch: Batch size

    Returns:
        Dict mapping tensor names to shapes
    """
    # Flux uses 16x16 patches, latent is 8x downsampled from pixel space
    # So for 1024x1024: latent is 128x128, patches are 8x8 = 64x64 = 4096 tokens
    latent_h = height // 8
    latent_w = width // 8
    patch_h = latent_h // 2  # Flux uses 2x2 patches on latent
    patch_w = latent_w // 2
    img_tokens = patch_h * patch_w

    # Text tokens are typically 512 for Flux (T5 max length)
    txt_tokens = 512

    # Hidden dimension is 3072 for Flux
    hidden_dim = 3072

    return {
        "hidden_states": (batch, img_tokens, hidden_dim),
        "encoder_hidden_states": (batch, txt_tokens, hidden_dim),
        "temb": (batch, hidden_dim),
        "gate": (batch, hidden_dim),
        "img_tokens": img_tokens,
        "txt_tokens": txt_tokens,
        "hidden_dim": hidden_dim,
    }
