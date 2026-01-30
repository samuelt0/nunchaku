"""
Capture tensor shapes at kernel boundaries during model inference.

This module provides utilities to extract model dimensions and calculate
accurate memory bandwidth based on actual tensor shapes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch


@dataclass
class KernelCall:
    """Record of a single kernel invocation."""
    kernel_name: str
    shapes: Dict[str, Tuple[int, ...]]
    dtypes: Dict[str, torch.dtype]
    bytes_read: int
    bytes_written: int

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written


@dataclass
class FluxModelConfig:
    """Configuration extracted from a Flux model.

    Attributes:
        hidden_dim: Model hidden dimension (typically 3072)
        mlp_hidden_dim: FFN hidden dimension (typically 4x hidden_dim)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_double_blocks: Number of double transformer blocks (19 for Flux)
        num_single_blocks: Number of single transformer blocks (38 for Flux)
        img_tokens: Number of image tokens (depends on resolution)
        txt_tokens: Number of text tokens (typically 512)
        dtype_bytes: Bytes per element (2 for bf16/fp16)
    """
    hidden_dim: int = 3072
    mlp_hidden_dim: int = 12288
    num_heads: int = 24
    head_dim: int = 128
    num_double_blocks: int = 19
    num_single_blocks: int = 38
    img_tokens: int = 4096
    txt_tokens: int = 512
    dtype_bytes: int = 2

    @classmethod
    def from_model(cls, model: torch.nn.Module, height: int = 1024, width: int = 1024) -> "FluxModelConfig":
        """Extract configuration from a loaded model.

        Args:
            model: NunchakuFluxTransformer2dModel or similar
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            FluxModelConfig with extracted dimensions
        """
        # Calculate image tokens from resolution (16x16 patches)
        img_tokens = (height // 16) * (width // 16)

        # Try to extract from model config
        config = getattr(model, "config", None)
        if config is not None:
            hidden_dim = getattr(config, "joint_attention_dim", 3072)
            num_heads = getattr(config, "num_attention_heads", 24)
            num_double_blocks = len(getattr(model, "transformer_blocks", [])) or 19
            num_single_blocks = len(getattr(model, "single_transformer_blocks", [])) or 38
        else:
            hidden_dim = 3072
            num_heads = 24
            num_double_blocks = 19
            num_single_blocks = 38

        head_dim = hidden_dim // num_heads
        mlp_hidden_dim = hidden_dim * 4  # Flux uses 4x expansion

        return cls(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_double_blocks=num_double_blocks,
            num_single_blocks=num_single_blocks,
            img_tokens=img_tokens,
            txt_tokens=512,
            dtype_bytes=2,
        )

    @property
    def combined_tokens(self) -> int:
        return self.img_tokens + self.txt_tokens


def calculate_flux_kernel_bytes(
    config: FluxModelConfig,
    num_steps: int = 50,
) -> Dict[str, Dict[str, int]]:
    """Calculate exact bytes for each kernel type based on Flux architecture.

    This function computes memory traffic based on the actual Flux model structure,
    accounting for different tensor shapes in different operations.

    Args:
        config: FluxModelConfig with model dimensions
        num_steps: Number of diffusion steps

    Returns:
        Dict with kernel names mapping to {bytes, calls} per step and total
    """
    h = config.hidden_dim
    mlp_h = config.mlp_hidden_dim
    img_t = config.img_tokens
    txt_t = config.txt_tokens
    combined_t = config.combined_tokens
    db = 2  # dtype bytes (bf16)

    results = {
        "quantize_w4a4_fuse_lora": {"bytes_per_step": 0, "calls_per_step": 0},
        "generalLayerNorm": {"bytes_per_step": 0, "calls_per_step": 0},
        "mul_add_kernel": {"bytes_per_step": 0, "calls_per_step": 0},
        "add_kernel": {"bytes_per_step": 0, "calls_per_step": 0},
    }

    # =========================================================================
    # Double blocks (19 blocks)
    # =========================================================================
    for _ in range(config.num_double_blocks):
        # ----- LayerNorm -----
        # norm1 (img): [batch, img_tokens, hidden_dim] - read + write
        results["generalLayerNorm"]["bytes_per_step"] += img_t * h * db * 2
        results["generalLayerNorm"]["calls_per_step"] += 1
        # norm1_context (txt): [batch, txt_tokens, hidden_dim]
        results["generalLayerNorm"]["bytes_per_step"] += txt_t * h * db * 2
        results["generalLayerNorm"]["calls_per_step"] += 1
        # norm2 (img)
        results["generalLayerNorm"]["bytes_per_step"] += img_t * h * db * 2
        results["generalLayerNorm"]["calls_per_step"] += 1
        # norm2_context (txt)
        results["generalLayerNorm"]["bytes_per_step"] += txt_t * h * db * 2
        results["generalLayerNorm"]["calls_per_step"] += 1

        # ----- Quantize (before GEMM) -----
        # QKV img: [img_tokens, hidden_dim] -> [img_tokens, hidden_dim*3]
        # Read bf16, write int4 packed (0.5 bytes) + scales
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += img_t * h * db + img_t * h * 3 // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # QKV txt: [txt_tokens, hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += txt_t * h * db + txt_t * h * 3 // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # Out proj img: [img_tokens, hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += img_t * h * db + img_t * h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # Out proj txt: [txt_tokens, hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += txt_t * h * db + txt_t * h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # FF1 img: [img_tokens, hidden_dim] -> [img_tokens, mlp_hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += img_t * h * db + img_t * mlp_h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # FF2 img: [img_tokens, mlp_hidden_dim] -> [img_tokens, hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += img_t * mlp_h * db + img_t * h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # FF1 txt
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += txt_t * h * db + txt_t * mlp_h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # FF2 txt
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += txt_t * mlp_h * db + txt_t * h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1

        # ----- mul_add (scale + shift after norm, gate operations) -----
        # scale+shift after norm1 img: x * scale + shift
        results["mul_add_kernel"]["bytes_per_step"] += img_t * h * db * 2  # read x, write x
        results["mul_add_kernel"]["calls_per_step"] += 1
        # scale+shift after norm1 txt
        results["mul_add_kernel"]["bytes_per_step"] += txt_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # scale+shift after norm2 img
        results["mul_add_kernel"]["bytes_per_step"] += img_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # scale+shift after norm2 txt
        results["mul_add_kernel"]["bytes_per_step"] += txt_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # gate_msa img: attn_out * gate
        results["mul_add_kernel"]["bytes_per_step"] += img_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # gate_msa txt
        results["mul_add_kernel"]["bytes_per_step"] += txt_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # gate_mlp img: ff_out * gate
        results["mul_add_kernel"]["bytes_per_step"] += img_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # gate_mlp txt
        results["mul_add_kernel"]["bytes_per_step"] += txt_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1

        # ----- add (residual connections) -----
        # residual add img (attn): hidden + attn_out
        results["add_kernel"]["bytes_per_step"] += img_t * h * db * 3  # read 2, write 1
        results["add_kernel"]["calls_per_step"] += 1
        # residual add img (ff)
        results["add_kernel"]["bytes_per_step"] += img_t * h * db * 3
        results["add_kernel"]["calls_per_step"] += 1
        # residual add txt (attn)
        results["add_kernel"]["bytes_per_step"] += txt_t * h * db * 3
        results["add_kernel"]["calls_per_step"] += 1
        # residual add txt (ff)
        results["add_kernel"]["bytes_per_step"] += txt_t * h * db * 3
        results["add_kernel"]["calls_per_step"] += 1

    # =========================================================================
    # Single blocks (38 blocks) - operate on concatenated img+txt
    # =========================================================================
    for _ in range(config.num_single_blocks):
        # ----- LayerNorm -----
        # norm: [batch, combined_tokens, hidden_dim]
        results["generalLayerNorm"]["bytes_per_step"] += combined_t * h * db * 2
        results["generalLayerNorm"]["calls_per_step"] += 1

        # ----- Quantize -----
        # QKV: [combined_tokens, hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += combined_t * h * db + combined_t * h * 3 // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # MLP fc1: [combined_tokens, hidden_dim] -> [combined_tokens, mlp_hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += combined_t * h * db + combined_t * mlp_h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1
        # proj_out (shared for attn + mlp): [combined_tokens, hidden_dim + mlp_hidden_dim]
        results["quantize_w4a4_fuse_lora"]["bytes_per_step"] += combined_t * (h + mlp_h) * db + combined_t * h // 2
        results["quantize_w4a4_fuse_lora"]["calls_per_step"] += 1

        # ----- mul_add -----
        # scale+shift after norm
        results["mul_add_kernel"]["bytes_per_step"] += combined_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1
        # gate: (attn + mlp) * gate
        results["mul_add_kernel"]["bytes_per_step"] += combined_t * h * db * 2
        results["mul_add_kernel"]["calls_per_step"] += 1

        # ----- add -----
        # attn + mlp (fused in single block)
        results["add_kernel"]["bytes_per_step"] += combined_t * h * db * 3
        results["add_kernel"]["calls_per_step"] += 1

    # =========================================================================
    # Final operations
    # =========================================================================
    # Final norm_out
    results["generalLayerNorm"]["bytes_per_step"] += img_t * h * db * 2
    results["generalLayerNorm"]["calls_per_step"] += 1

    # Scale by number of steps
    for kernel in results:
        results[kernel]["total_bytes"] = results[kernel]["bytes_per_step"] * num_steps
        results[kernel]["total_calls"] = results[kernel]["calls_per_step"] * num_steps

    return results


@dataclass
class KernelShapeCapture:
    """Captures tensor shapes for kernel calls during inference.

    Usage:
        capture = KernelShapeCapture()
        capture.install_hooks()

        # Run inference
        model(inputs)

        capture.remove_hooks()

        # Get captured data
        summary = capture.get_summary()
    """

    calls: List[KernelCall] = field(default_factory=list)
    _hooks: List[Any] = field(default_factory=list)
    _original_fns: Dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True

    def clear(self):
        """Clear all captured data."""
        self.calls.clear()

    def _record_quantize(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        oscales: torch.Tensor,
        lora_down: Optional[torch.Tensor],
        lora_act_out: Optional[torch.Tensor],
        **kwargs,
    ):
        """Record a quantize kernel call."""
        if not self.enabled:
            return

        M, K = input.shape
        dtype_bytes = input.element_size()

        # Bytes read: input (BF16)
        bytes_read = M * K * dtype_bytes

        # Bytes written: output (INT4 packed = K/2 bytes per row) + oscales
        bytes_written = output.numel() * output.element_size()
        bytes_written += oscales.numel() * oscales.element_size()
        if lora_act_out is not None:
            bytes_written += lora_act_out.numel() * lora_act_out.element_size()

        self.calls.append(KernelCall(
            kernel_name="quantize_w4a4_fuse_lora",
            shapes={
                "input": tuple(input.shape),
                "output": tuple(output.shape),
                "oscales": tuple(oscales.shape),
            },
            dtypes={
                "input": input.dtype,
                "output": output.dtype,
            },
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        ))

    def _record_layernorm(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        **kwargs,
    ):
        """Record a layernorm kernel call."""
        if not self.enabled:
            return

        dtype_bytes = input.element_size()
        numel = input.numel()

        # Read input, write output
        bytes_read = numel * dtype_bytes
        bytes_written = numel * dtype_bytes

        self.calls.append(KernelCall(
            kernel_name="generalLayerNorm",
            shapes={"input": tuple(input.shape)},
            dtypes={"input": input.dtype},
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        ))

    def _record_mul_add(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        **kwargs,
    ):
        """Record a mul_add kernel call (x = x * scale + shift)."""
        if not self.enabled:
            return

        dtype_bytes = x.element_size()
        x_numel = x.numel()
        scale_numel = scale.numel()

        # Read x, scale, shift; write x (in-place)
        bytes_read = x_numel * dtype_bytes + scale_numel * dtype_bytes * 2
        bytes_written = x_numel * dtype_bytes

        self.calls.append(KernelCall(
            kernel_name="mul_add_kernel",
            shapes={
                "x": tuple(x.shape),
                "scale": tuple(scale.shape),
            },
            dtypes={"x": x.dtype},
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        ))

    def _record_add(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Record an add kernel call (c = a + b)."""
        if not self.enabled:
            return

        dtype_bytes = a.element_size()
        numel = a.numel()

        # Read a, b; write c
        bytes_read = numel * dtype_bytes * 2
        bytes_written = numel * dtype_bytes

        self.calls.append(KernelCall(
            kernel_name="add_kernel",
            shapes={"a": tuple(a.shape), "b": tuple(b.shape)},
            dtypes={"a": a.dtype},
            bytes_read=bytes_read,
            bytes_written=bytes_written,
        ))

    def install_hooks(self):
        """Install hooks to capture kernel shapes.

        Hooks into:
        - nunchaku.ops.quantize.svdq_quantize_w4a4_act_fuse_lora_cuda
        - torch.nn.LayerNorm.forward
        - torch mul/add operations in normalization
        """
        from nunchaku.ops import quantize as quantize_module

        # Hook quantize function
        original_quantize = quantize_module.svdq_quantize_w4a4_act_fuse_lora_cuda
        self._original_fns["quantize"] = original_quantize

        def hooked_quantize(input, output=None, oscales=None, lora_down=None,
                          lora_act_out=None, smooth=None, fuse_glu=False,
                          fp4=False, pad_size=256):
            result = original_quantize(input, output, oscales, lora_down,
                                       lora_act_out, smooth, fuse_glu, fp4, pad_size)
            self._record_quantize(input, result[0], result[1], lora_down, result[2])
            return result

        quantize_module.svdq_quantize_w4a4_act_fuse_lora_cuda = hooked_quantize

        # Hook LayerNorm - we'll use a forward hook on the module
        # This will be installed on specific model instances

    def install_model_hooks(self, model: torch.nn.Module):
        """Install hooks on a specific model instance.

        Args:
            model: The model to hook (e.g., NunchakuFluxTransformer2dModel)
        """
        self.install_hooks()

        # Hook all LayerNorm modules
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                def make_hook(mod_name):
                    def hook(mod, args, output):
                        input_tensor = args[0]
                        self._record_layernorm(input_tensor, output)
                    return hook
                handle = module.register_forward_hook(make_hook(name))
                self._hooks.append(handle)

        # Hook the transformer blocks for mul_add and add operations
        # We'll track tensor operations through custom hooks
        self._install_block_hooks(model)

    def _install_block_hooks(self, model: torch.nn.Module):
        """Install hooks on transformer blocks to capture mul_add and add ops."""
        from nunchaku.models.transformers.transformer_flux_v2 import (
            NunchakuFluxTransformerBlock,
            NunchakuFluxSingleTransformerBlock,
        )

        for name, module in model.named_modules():
            if isinstance(module, NunchakuFluxTransformerBlock):
                self._hook_double_block(module, name)
            elif isinstance(module, NunchakuFluxSingleTransformerBlock):
                self._hook_single_block(module, name)

    def _hook_double_block(self, block, name: str):
        """Hook a double transformer block."""
        original_forward = block.forward

        def hooked_forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            # Track shapes before forward
            hidden_shape = hidden_states.shape
            encoder_shape = encoder_hidden_states.shape
            dtype_bytes = hidden_states.element_size()

            # Run original forward
            result = original_forward(
                hidden_states, encoder_hidden_states, temb,
                image_rotary_emb, joint_attention_kwargs
            )

            # Record mul_add operations (scale + shift after norm)
            # Double block has: 2x scale+shift for norm1 (img, txt), 2x for norm2 (img, txt)
            # Plus gate operations: 2x gate_msa, 2x gate_mlp
            for _ in range(4):  # 4 scale+shift operations
                self._record_mul_add(
                    torch.empty(hidden_shape, dtype=hidden_states.dtype),
                    torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
                    torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
                )
            for _ in range(2):  # 2 gate operations for img
                self._record_mul_add(
                    torch.empty(hidden_shape, dtype=hidden_states.dtype),
                    torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
                    torch.empty(0, dtype=hidden_states.dtype),  # no shift for gate
                )
            for _ in range(2):  # 2 gate operations for txt
                self._record_mul_add(
                    torch.empty(encoder_shape, dtype=hidden_states.dtype),
                    torch.empty(encoder_shape[0], encoder_shape[2], dtype=hidden_states.dtype),
                    torch.empty(0, dtype=hidden_states.dtype),
                )

            # Record add operations (residuals)
            # img: attn_output + residual, ff_output + residual
            # txt: attn_output + residual, ff_output + residual
            for _ in range(2):  # 2 adds for img
                self._record_add(
                    torch.empty(hidden_shape, dtype=hidden_states.dtype),
                    torch.empty(hidden_shape, dtype=hidden_states.dtype),
                )
            for _ in range(2):  # 2 adds for txt
                self._record_add(
                    torch.empty(encoder_shape, dtype=hidden_states.dtype),
                    torch.empty(encoder_shape, dtype=hidden_states.dtype),
                )

            return result

        block.forward = hooked_forward

    def _hook_single_block(self, block, name: str):
        """Hook a single transformer block."""
        original_forward = block.forward

        def hooked_forward(
            hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            hidden_shape = hidden_states.shape
            dtype_bytes = hidden_states.element_size()

            result = original_forward(
                hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )

            # Record mul_add: 1x scale+shift, 1x gate
            self._record_mul_add(
                torch.empty(hidden_shape, dtype=hidden_states.dtype),
                torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
                torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
            )
            self._record_mul_add(
                torch.empty(hidden_shape, dtype=hidden_states.dtype),
                torch.empty(hidden_shape[0], hidden_shape[2], dtype=hidden_states.dtype),
                torch.empty(0, dtype=hidden_states.dtype),
            )

            # Record add: attn+mlp, residual
            self._record_add(
                torch.empty(hidden_shape, dtype=hidden_states.dtype),
                torch.empty(hidden_shape, dtype=hidden_states.dtype),
            )

            return result

        block.forward = hooked_forward

    def remove_hooks(self):
        """Remove all installed hooks."""
        # Remove forward hooks
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Restore original functions
        if "quantize" in self._original_fns:
            from nunchaku.ops import quantize as quantize_module
            quantize_module.svdq_quantize_w4a4_act_fuse_lora_cuda = self._original_fns["quantize"]

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics grouped by kernel type.

        Returns:
            Dict with kernel names as keys, containing:
            - count: number of calls
            - total_bytes_read: sum of bytes read
            - total_bytes_written: sum of bytes written
            - total_bytes: total bytes (read + write)
            - shapes: list of unique shapes seen
        """
        summary = {}

        for call in self.calls:
            if call.kernel_name not in summary:
                summary[call.kernel_name] = {
                    "count": 0,
                    "total_bytes_read": 0,
                    "total_bytes_written": 0,
                    "total_bytes": 0,
                    "shapes": set(),
                }

            s = summary[call.kernel_name]
            s["count"] += 1
            s["total_bytes_read"] += call.bytes_read
            s["total_bytes_written"] += call.bytes_written
            s["total_bytes"] += call.total_bytes

            # Record unique shapes
            shape_key = str(call.shapes)
            s["shapes"].add(shape_key)

        # Convert sets to lists for serialization
        for kernel in summary:
            summary[kernel]["shapes"] = list(summary[kernel]["shapes"])

        return summary

    def get_bytes_by_kernel(self) -> Dict[str, int]:
        """Get total bytes transferred per kernel type.

        Returns:
            Dict mapping kernel name to total bytes
        """
        summary = self.get_summary()
        return {k: v["total_bytes"] for k, v in summary.items()}

    def get_call_counts(self) -> Dict[str, int]:
        """Get call counts per kernel type.

        Returns:
            Dict mapping kernel name to call count
        """
        summary = self.get_summary()
        return {k: v["count"] for k, v in summary.items()}


def capture_kernel_shapes(
    model: torch.nn.Module,
    forward_fn: Callable,
    *args,
    **kwargs,
) -> KernelShapeCapture:
    """Convenience function to capture kernel shapes during a single forward pass.

    Args:
        model: The model to profile
        forward_fn: Function to call for forward pass (e.g., model.forward)
        *args, **kwargs: Arguments to pass to forward_fn

    Returns:
        KernelShapeCapture with recorded data
    """
    capture = KernelShapeCapture()
    capture.install_model_hooks(model)

    try:
        with torch.no_grad():
            forward_fn(*args, **kwargs)
    finally:
        capture.remove_hooks()

    return capture
