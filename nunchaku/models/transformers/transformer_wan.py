"""
Implements the NunchakuWanTransformer3DModel, a quantized Wan transformer for Diffusers with efficient inference support.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanTransformerBlock,
    WanTransformer3DModel,
)
from huggingface_hub import utils
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..normalization import FP32LayerNorm  # Now properly available
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin


class NunchakuWanAttention(NunchakuBaseAttention):
    """
    Quantized Wan attention module using SVDQW4A4Linear for efficient inference.
    """
    
    def __init__(self, other: WanAttention, processor: str = "nunchaku-fp16", **kwargs):
        super(NunchakuWanAttention, self).__init__(processor)
        
        self.inner_dim = other.inner_dim
        self.heads = other.heads
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.cross_attention_dim_head = other.cross_attention_dim_head
        self.kv_inner_dim = other.kv_inner_dim
        
        # Copy normalization layers
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        
        # Fuse and quantize QKV projections for self-attention
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        
        # Output projection
        self.to_out = nn.ModuleList([
            SVDQW4A4Linear.from_linear(other.to_out[0], **kwargs),
            other.to_out[1]  # Dropout layer
        ])
        
        # Handle additional KV projections for I2V if present
        if self.added_kv_proj_dim is not None:
            if hasattr(other, 'add_k_proj') and hasattr(other, 'add_v_proj'):
                with torch.device("meta"):
                    add_kv = fuse_linears([other.add_k_proj, other.add_v_proj])
                self.add_kv_proj = SVDQW4A4Linear.from_linear(add_kv, **kwargs)
                self.norm_added_k = other.norm_added_k
            else:
                self.add_kv_proj = None
                self.norm_added_k = None
        else:
            self.add_kv_proj = None
            self.norm_added_k = None
        
        self.is_cross_attention = other.is_cross_attention
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for quantized Wan attention."""
        
        # Handle cross-attention vs self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        # Check for image encoder states (I2V)
        encoder_hidden_states_img = None
        if self.add_kv_proj is not None and encoder_hidden_states is not None:
            # Split image and text context if needed
            # 512 is the context length of the text encoder (hardcoded for now)
            if encoder_hidden_states.shape[1] > 512:
                image_context_length = encoder_hidden_states.shape[1] - 512
                encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
                encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        
        # Get QKV projections
        qkv = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply normalization
        q = self.norm_q(q)
        k = self.norm_k(k)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            q = self._apply_rotary_emb(q, rotary_emb)
            k = self._apply_rotary_emb(k, rotary_emb)
        
        # Handle image KV projections if present
        hidden_states_img = None
        if encoder_hidden_states_img is not None and self.add_kv_proj is not None:
            kv_img = self.add_kv_proj(encoder_hidden_states_img)
            k_img, v_img = kv_img.chunk(2, dim=-1)
            k_img = self.norm_added_k(k_img)
            
            k_img = k_img.view(batch_size, -1, self.heads, k_img.shape[-1] // self.heads).transpose(1, 2)
            v_img = v_img.view(batch_size, -1, self.heads, v_img.shape[-1] // self.heads).transpose(1, 2)
            
            # Compute attention with image KV
            hidden_states_img = F.scaled_dot_product_attention(
                q, k_img, v_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).contiguous()
            hidden_states_img = hidden_states_img.view(batch_size, seq_len, -1)
        
        # Compute main attention
        hidden_states = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size, seq_len, -1)
        
        # Add image attention if computed
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        
        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)  # Dropout
        
        return hidden_states
    
    def _apply_rotary_emb(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Apply rotary embeddings to hidden states."""
        freqs_cos, freqs_sin = rotary_emb
        
        # Reshape for rotary application
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        
        return out.type_as(hidden_states)


class NunchakuWanTransformerBlock(WanTransformerBlock):
    """
    Quantized Wan transformer block with SVDQW4A4Linear layers.
    """
    
    def __init__(self, block: WanTransformerBlock, **kwargs):
        super(WanTransformerBlock, self).__init__()
        
        # Copy configuration
        self.scale_shift_table = block.scale_shift_table
        
        # Normalization layers
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.norm3 = block.norm3
        
        # Quantized attention layers
        self.attn1 = NunchakuWanAttention(block.attn1, **kwargs)
        self.attn2 = NunchakuWanAttention(block.attn2, **kwargs)
        
        # Quantized feed-forward network
        self.ffn = NunchakuFeedForward(block.ffn, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the quantized transformer block."""
        
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
        
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output
        
        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        
        return hidden_states


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    """
    WanTransformer3DModel with Nunchaku quantized backend support.
    
    This class extends the base WanTransformer3DModel to support loading and
    using quantized transformer blocks with pure Python implementation.
    """
    
    def _patch_model(self, **kwargs):
        """Replace model components with quantized versions."""
        # Replace transformer blocks with quantized versions
        for i, block in enumerate(self.blocks):
            self.blocks[i] = NunchakuWanTransformerBlock(block, **kwargs)
        return self
    
    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuWanTransformer3DModel from a local file or HuggingFace Hub.
        
        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the model checkpoint or HuggingFace Hub model name.
        **kwargs
            Additional keyword arguments for model loading.
            
        Returns
        -------
        NunchakuWanTransformer3DModel
            The loaded model with quantized blocks.
        """
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        
        # Check if it's a single safetensors file or a directory
        if pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ):
            # Load from single file
            transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
            quantization_config = json.loads(metadata.get("quantization_config", "{}"))
            rank = quantization_config.get("rank", 32)
        else:
            # Load from directory with separate files
            transformer_blocks_path = pretrained_model_name_or_path / "transformer_blocks.safetensors"
            unquantized_layers_path = pretrained_model_name_or_path / "unquantized_layers.safetensors"
            
            if not transformer_blocks_path.exists() or not unquantized_layers_path.exists():
                raise ValueError(
                    f"Expected quantized model files not found in {pretrained_model_name_or_path}. "
                    "Make sure you have converted the model using convert.py first."
                )
            
            # Load the base model structure
            transformer = cls(
                **kwargs.get("config", {})
            )
            
            # Load quantized blocks
            quantized_state_dict = load_file(transformer_blocks_path)
            unquantized_state_dict = load_file(unquantized_layers_path)
            
            model_state_dict = {**quantized_state_dict, **unquantized_state_dict}
            rank = 32  # Default rank, can be overridden
        
        # Convert to specified dtype
        transformer = transformer.to(torch_dtype)
        
        # Patch the model with quantized components
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank)
        
        # Move to device and load weights
        transformer = transformer.to_empty(device=device)
        
        # Convert state dict format if needed
        converted_state_dict = convert_wan_state_dict(model_state_dict)
        
        # Load the weights
        transformer.load_state_dict(converted_state_dict, strict=False)
        
        return transformer


def convert_wan_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Wan state dict from convert.py format to the format expected by NunchakuWanTransformer3DModel.
    """
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_k = k
        
        # Map quantized weight names
        if "blocks." in k:
            # Map attention projections (self-attention)
            if ".qkv_proj." in k and "context" not in k:
                new_k = k.replace(".qkv_proj.", ".attn1.to_qkv.")
            # Map attention projections (cross-attention)
            elif ".qkv_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn2.to_qkv.")
            # Map additional KV projections (I2V)
            elif ".add_kv_proj." in k:
                new_k = k.replace(".add_kv_proj.", ".attn2.add_kv_proj.")
            # Map output projections
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn2.to_out.0.")
            elif ".out_proj." in k and "context" not in k:
                new_k = k.replace(".out_proj.", ".attn1.to_out.0.")
            
            # Map FFN projections
            elif ".mlp_fc1." in k:
                new_k = k.replace(".mlp_fc1.", ".ffn.net.0.proj.")
            elif ".mlp_fc2." in k:
                new_k = k.replace(".mlp_fc2.", ".ffn.net.2.")
            
            # Map normalization layers more explicitly
            # Self-attention norms
            elif ".norm_q." in k and "context" not in k:
                new_k = k.replace(".norm_q.", ".attn1.norm_q.")
            elif ".norm_k." in k and "context" not in k and "added" not in k:
                new_k = k.replace(".norm_k.", ".attn1.norm_k.")
            # Cross-attention norms
            elif ".norm_q_context." in k:
                new_k = k.replace(".norm_q_context.", ".attn2.norm_q.")
            elif ".norm_k_context." in k:
                new_k = k.replace(".norm_k_context.", ".attn2.norm_k.")
            # Additional KV norm (I2V)
            elif ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn2.norm_added_k.")
            
            # Map LoRA weights
            if ".lora_down" in k:
                new_k = new_k.replace(".lora_down", ".proj_down")
            elif ".lora_up" in k:
                new_k = new_k.replace(".lora_up", ".proj_up")
            
            # Map smooth factors
            if ".smooth_orig" in k:
                new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in k and ".smooth_factor" not in k:
                new_k = new_k.replace(".smooth", ".smooth_factor")
        
        new_state_dict[new_k] = v
    
    return new_state_dict
