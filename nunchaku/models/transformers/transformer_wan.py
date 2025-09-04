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
from diffusers.models.attention_dispatch import dispatch_attention_fn
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
        
        # Determine if we're dealing with self-attention or cross-attention
        # Self-attention: all three (Q, K, V) have same dimension
        # Cross-attention: Q has different dimension from K, V
        self.is_cross_attention = other.is_cross_attention
        
        # Use fused QKV for self-attention (like NunchakuFluxTransformer)
        if not self.is_cross_attention:
            # Self-attention: fuse Q, K, V into a single projection
            with torch.device("meta"):
                to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
            self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
            self.to_q = None
            self.to_k = None
            self.to_v = None
        else:
            # Cross-attention: handle mixed quantization
            # Q is quantized, but K and V might not be
            self.to_q = SVDQW4A4Linear.from_linear(other.to_q, **kwargs)
            # Keep K and V separate (they are not quantized in the converted checkpoint)
            self.to_k = other.to_k  # Keep as regular Linear (not quantized)
            self.to_v = other.to_v  # Keep as regular Linear (not quantized)
            self.to_qkv = None
        
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
    
    def set_processor(self, processor: str):
        """
        Set the attention processor type.
        
        Parameters
        ----------
        processor : str
            The processor type to use for attention computation.
        """
        self.processor = processor
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for quantized Wan attention."""
        
        # Check for image encoder states (I2V)
        encoder_hidden_states_img = None
        if self.add_kv_proj is not None and encoder_hidden_states is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        
        # Get QKV projections based on attention type
        if not self.is_cross_attention:
            # Self-attention: use fused QKV
            qkv = self.to_qkv(hidden_states)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross-attention: Q from hidden_states, K and V from encoder_hidden_states
            q = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
        
        # Apply normalization
        q = self.norm_q(q)
        k = self.norm_k(k)
        
        # Reshape for multi-head attention - using unflatten to match original implementation
        q = q.unflatten(2, (self.heads, -1))  # [batch, seq_len, heads, head_dim]
        k = k.unflatten(2, (self.heads, -1))  # [batch, seq_len, heads, head_dim]
        v = v.unflatten(2, (self.heads, -1))  # [batch, seq_len, heads, head_dim]
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            q = self._apply_rotary_emb(q, rotary_emb)
            k = self._apply_rotary_emb(k, rotary_emb)
        
        # Handle image KV projections if present (I2V task)
        hidden_states_img = None
        if encoder_hidden_states_img is not None and self.add_kv_proj is not None:
            kv_img = self.add_kv_proj(encoder_hidden_states_img)
            k_img, v_img = kv_img.chunk(2, dim=-1)
            k_img = self.norm_added_k(k_img)
            
            k_img = k_img.unflatten(2, (self.heads, -1))
            v_img = v_img.unflatten(2, (self.heads, -1))
            
            # Use dispatch_attention_fn to match original implementation
            hidden_states_img = dispatch_attention_fn(
                q,
                k_img,
                v_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=None
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(q)
        
        # Compute main attention using dispatch_attention_fn
        hidden_states = dispatch_attention_fn(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(q)
        
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
        """Apply rotary embeddings to hidden states.
        
        hidden_states: [batch, seq_len, heads, head_dim]
        rotary_emb: (freqs_cos, freqs_sin)
        """
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
    
    @staticmethod
    def _map_config_parameters(config_dict: dict) -> dict:
        """
        Map config parameters from the saved format to the expected format.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary from config.json
            
        Returns
        -------
        dict
            Mapped configuration dictionary for WanTransformer3DModel
        """
        # Create a new dict with mapped parameters
        mapped_config = {}
        
        # Map the parameters
        if 'dim' in config_dict and 'num_heads' in config_dict:
            mapped_config['num_attention_heads'] = config_dict['num_heads']
            mapped_config['attention_head_dim'] = config_dict['dim'] // config_dict['num_heads']
        
        if 'in_dim' in config_dict:
            mapped_config['in_channels'] = config_dict['in_dim']
        
        if 'out_dim' in config_dict:
            mapped_config['out_channels'] = config_dict['out_dim']
        
        if 'ffn_dim' in config_dict:
            mapped_config['ffn_dim'] = config_dict['ffn_dim']
        
        if 'freq_dim' in config_dict:
            mapped_config['freq_dim'] = config_dict['freq_dim']
        
        if 'num_layers' in config_dict:
            mapped_config['num_layers'] = config_dict['num_layers']
        
        if 'eps' in config_dict:
            mapped_config['eps'] = config_dict['eps']
        
        # For text_dim, we need to use a default value since text_len is not the same
        # text_dim should be the text encoder's embedding dimension (typically 4096 for T5)
        mapped_config['text_dim'] = config_dict.get('text_dim', 4096)
        
        return mapped_config
    
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
                # Try looking for merged safetensors file
                merged_path = pretrained_model_name_or_path / "merged.safetensors"
                if merged_path.exists():
                    # Load from merged file
                    transformer, model_state_dict, metadata = cls._build_model(merged_path, **kwargs)
                    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
                    rank = quantization_config.get("rank", 32)
                else:
                    raise ValueError(
                        f"Expected quantized model files not found in {pretrained_model_name_or_path}. "
                        "Make sure you have converted the model using convert.py first."
                    )
            else:
                # Load quantized blocks first to detect rank
                quantized_state_dict = load_file(transformer_blocks_path)
                unquantized_state_dict = load_file(unquantized_layers_path)
                
                model_state_dict = {**quantized_state_dict, **unquantized_state_dict}
                
                # Infer rank from the tensor shapes BEFORE creating the model
                # Look for wscales tensor to determine the actual rank used
                rank = 32  # Default fallback
                for key in quantized_state_dict.keys():
                    if "wscales" in key and "blocks.0.qkv_proj.wscales" in key:
                        # Expected shape for rank=32: [320, X]
                        # Actual shape for rank=8: [80, X]
                        actual_shape = quantized_state_dict[key].shape[0]
                        # For attention layers: expected = num_heads * attention_head_dim * 3 / rank
                        # num_heads=40, attention_head_dim=128, so inner_dim=5120
                        # For QKV: 5120 * 3 = 15360
                        # Expected groups for rank=32: 15360 / 48 = 320
                        # for rank=8, we get 80 groups (which matches)
                        # So rank = 32 * (80 / 320) = 8
                        expected_for_rank32 = 320
                        rank = int(32 * (actual_shape / expected_for_rank32))
                        print(f"Inferred rank={rank} from tensor shapes")
                        break
                
                # NOW create the model structure
                config_path = pretrained_model_name_or_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    # Map the config parameters to the expected format
                    mapped_config = cls._map_config_parameters(config_dict)
                    transformer = cls(**mapped_config)
                else:
                    transformer = cls(
                        **kwargs.get("config", {})
                    )
        
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
    Like NunchakuFluxTransformer, we keep the fused QKV tensors as-is.
    """
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_k = k
        
        # Map quantized weight names
        if "blocks." in k:
            # Map attention projections (self-attention) - keep fused QKV
            if ".qkv_proj." in k and "context" not in k:
                new_k = k.replace(".qkv_proj.", ".attn1.to_qkv.")
            # Map attention projections (cross-attention)
            # Handle indexed cross-attention projections from convert.py
            elif ".qkv_proj_context_0." in k or ".q_proj_context." in k:
                # Q projection (quantized)
                new_k = k.replace(".qkv_proj_context_0.", ".attn2.to_q.")
                new_k = new_k.replace(".q_proj_context.", ".attn2.to_q.")
            elif ".qkv_proj_context_1." in k or ".k_proj_context." in k:
                # K projection (unquantized)
                new_k = k.replace(".qkv_proj_context_1.", ".attn2.to_k.")
                new_k = new_k.replace(".k_proj_context.", ".attn2.to_k.")
            elif ".qkv_proj_context_2." in k or ".v_proj_context." in k:
                # V projection (unquantized)
                new_k = k.replace(".qkv_proj_context_2.", ".attn2.to_v.")
                new_k = new_k.replace(".v_proj_context.", ".attn2.to_v.")
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
