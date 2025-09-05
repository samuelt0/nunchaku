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
from typing import Any, Dict, Optional, Tuple, Union
from ...utils import get_precision
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..linear import SVDQW4A4Linear
from ..linear_padded import SVDQW4A4LinearPadded
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin


class NunchakuWanAttention(nn.Module):
    """
    Quantized Wan attention module using SVDQW4A4Linear for efficient inference.
    Properly handles mixed quantization where some layers remain unquantized.
    """
    
    def __init__(self, other: WanAttention, **kwargs):
        super().__init__()
        
        # Copy all attributes from the original attention
        self.heads = other.heads
        
        # Fix 3: Compute dim_head from norm shape instead of guessing
        norm_shape = other.norm_q.normalized_shape
        norm_total = norm_shape[0] if isinstance(norm_shape, (tuple, list)) else norm_shape
        self.dim_head = norm_total // other.heads
        self.inner_dim = norm_total
        
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.cross_attention_dim_head = getattr(other, 'cross_attention_dim_head', None)
        self.kv_inner_dim = getattr(other, 'kv_inner_dim', self.inner_dim)
        self.is_cross_attention = getattr(other, 'is_cross_attention', 
                                         self.cross_attention_dim_head is not None)
        
        # Copy normalization layers as-is
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        
        # Initialize fused_projections flag - will be set based on structure
        self.fused_projections = False
        
        # Handle projection structure based on what exists in the original model
        # For self-attention (attn1), we need to handle fused QKV
        if not self.is_cross_attention:
            # Self-attention - create fused QKV from separate projections if needed
            if hasattr(other, 'to_qkv') and other.to_qkv is not None:
                # Already fused
                self.to_qkv = SVDQW4A4Linear.from_linear(other.to_qkv, **kwargs)
                self.fused_projections = True
            elif hasattr(other, 'to_q') and other.to_q is not None:
                # Need to fuse Q, K, V for self-attention
                from ..utils import fuse_linears
                with torch.device("meta"):
                    fused_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
                self.to_qkv = SVDQW4A4Linear.from_linear(fused_qkv, **kwargs)
                self.fused_projections = True
            else:
                self.to_qkv = None
                
            # Clear separate projections for self-attention when fused
            if self.fused_projections:
                self.to_q = None
                self.to_k = None
                self.to_v = None
                self.to_kv = None
        else:
            # Fix 2: Cross-attention - ONLY Q is quantized; K/V stay float and NOT fused
            self.to_qkv = None
            
            # Q projection - quantized
            if hasattr(other, 'to_q') and other.to_q is not None:
                self.to_q = SVDQW4A4Linear.from_linear(other.to_q, **kwargs)
            else:
                self.to_q = None
            
            # Keep float K/V, do NOT fuse - processor will use them separately
            self.to_k = other.to_k  # Keep as regular Linear
            self.to_v = other.to_v  # Keep as regular Linear
            self.to_kv = None
            self.fused_projections = False  # Critical: must be False for separate K/V
        
        # Output projection - always quantized
        self.to_out = nn.ModuleList([
            SVDQW4A4Linear.from_linear(other.to_out[0], **kwargs),
            other.to_out[1]  # Dropout layer
        ])
        
        # Handle additional KV projections for I2V if present
        if self.added_kv_proj_dim is not None:
            # These remain unquantized based on checkpoint analysis
            self.add_k_proj = getattr(other, 'add_k_proj', None)
            self.add_v_proj = getattr(other, 'add_v_proj', None)
            self.to_added_kv = getattr(other, 'to_added_kv', None)
            self.norm_added_k = getattr(other, 'norm_added_k', None)
        else:
            self.to_added_kv = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.norm_added_k = None
        
        # Set the processor
        from diffusers.models.transformers.transformer_wan import WanAttnProcessor
        self.processor = WanAttnProcessor()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self, 
            hidden_states, 
            encoder_hidden_states, 
            attention_mask, 
            rotary_emb, 
            **kwargs
        )


class NunchakuWanTransformerBlock(WanTransformerBlock):
    """
    Quantized Wan transformer block with SVDQW4A4Linear layers.
    Minimal wrapper that only replaces necessary components.
    """
    
    def __init__(self, block: WanTransformerBlock, **kwargs):
        # Initialize without calling parent __init__
        super(WanTransformerBlock, self).__init__()
        
        # Initialize scale_shift_table properly
        # Check if it exists on the block (it might not be initialized yet)
        if hasattr(block, 'scale_shift_table') and block.scale_shift_table is not None:
            self.scale_shift_table = block.scale_shift_table
        else:
            # Create it ourselves based on the block's expected dimensions
            # Get dim from the norm1 layer's normalized shape
            norm_shape = block.norm1.normalized_shape
            dim = norm_shape[0] if isinstance(norm_shape, (tuple, list)) else norm_shape
            # Initialize as per the official WanTransformerBlock
            self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # Copy normalization layers as-is
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.norm3 = block.norm3
        
        # Replace attention modules with quantized versions
        self.attn1 = NunchakuWanAttention(block.attn1, **kwargs)
        self.attn2 = NunchakuWanAttention(block.attn2, **kwargs)
        
        # Replace feed-forward with quantized version
        self.ffn = NunchakuFeedForward(block.ffn, **kwargs)


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    """
    WanTransformer3DModel with Nunchaku quantized backend support.
    
    This class extends the base WanTransformer3DModel to support loading and
    using quantized transformer blocks. It maintains full compatibility with
    the diffusers implementation while using quantized linear layers.
    """
    
    def _patch_model(self, **kwargs):
        """Replace model components with quantized versions."""
        # Replace transformer blocks with quantized versions
        for i, block in enumerate(self.blocks):
            self.blocks[i] = NunchakuWanTransformerBlock(block, **kwargs)
        
        # Patch other linear layers if they exist
        # Note: WanTransformer3DModel typically doesn't have in_proj/out_proj
        # but we check just in case
        if hasattr(self, 'patch_embedding') and isinstance(self.patch_embedding, nn.Conv3d):
            # Keep Conv3d as-is, it's not a Linear layer
            pass
        
        if hasattr(self, 'proj_out') and isinstance(self.proj_out, nn.Linear):
            # Keep proj_out as a regular Linear layer since the checkpoint has regular weights for it
            # The checkpoint has proj_out.weight and proj_out.bias, not quantized weights
            pass  # Don't quantize proj_out
        
        # Fix 1: DO NOT quantize condition_embedder - it's critical for time/text/image projections
        # The condition embedder should remain unquantized as it's very sensitive to precision
        
        return self
    
    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuWanTransformer3DModel from a local file or HuggingFace Hub.
        
        This method follows the pattern established by NunchakuQwenImageTransformer2DModel
        for proper weight loading and quantization.
        """
        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        
        offload = kwargs.get("offload", False)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        
        # Use the _build_model method from NunchakuModelLoaderMixin
        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        
        # Extract quantization config and model config from metadata
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))
        rank = quantization_config.get("rank", 32)
        
        # Convert to specified dtype
        transformer = transformer.to(torch_dtype)
        
        # Determine precision
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)
        if precision == "fp4":
            precision = "nvfp4"
        
        # Patch the model with quantized components
        transformer._patch_model(precision=precision, rank=rank)
        
        # Move to device
        transformer = transformer.to_empty(device=device)
        
        # Handle wcscales if they're missing (like in QwenImage)
        state_dict = transformer.state_dict()
        for k in state_dict.keys():
            if k not in model_state_dict:
                if ".wcscales" in k:
                    model_state_dict[k] = torch.ones_like(state_dict[k])
        
        # CRITICAL FIX: Check if checkpoint is already converted
        def is_already_converted(state_dict):
            """Check if keys already have attn1/attn2/ffn structure"""
            return any('attn1.' in k or 'attn2.' in k or '.ffn.' in k for k in state_dict.keys())
        
        # Only convert if not already converted
        if not is_already_converted(model_state_dict):
            print("Converting state dict from original format...")
            converted_state_dict = convert_wan_state_dict(model_state_dict)
        else:
            print("Checkpoint is already converted, using directly...")
            converted_state_dict = model_state_dict  # Already converted!
        
        # Handle wtscale loading for SVDQW4A4Linear modules
        for n, m in transformer.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if hasattr(m, 'wtscale') and m.wtscale is not None:
                    wtscale_key = f"{n}.wtscale"
                    if wtscale_key in converted_state_dict:
                        m.wtscale = converted_state_dict.pop(wtscale_key, 1.0)
        
        # Handle padded layers - we need to pad the weights from checkpoint
        # before loading them into padded layers
        for n, m in transformer.named_modules():
            if isinstance(m, SVDQW4A4LinearPadded) and m.needs_padding:
                # Pad bias if it exists in the state dict
                bias_key = f"{n}.bias"
                if bias_key in converted_state_dict:
                    orig_bias = converted_state_dict[bias_key]
                    padded_bias = torch.zeros(m.out_features_padded, dtype=orig_bias.dtype, device=orig_bias.device)
                    padded_bias[:m.original_out_features] = orig_bias
                    converted_state_dict[bias_key] = padded_bias
                
                # Pad qweight if it exists (shape: [out_features, in_features // 2])
                qweight_key = f"{n}.qweight"
                if qweight_key in converted_state_dict:
                    orig_qweight = converted_state_dict[qweight_key]
                    padded_shape = list(orig_qweight.shape)
                    padded_shape[0] = m.out_features_padded
                    padded_qweight = torch.zeros(padded_shape, dtype=orig_qweight.dtype, device=orig_qweight.device)
                    padded_qweight[:orig_qweight.shape[0]] = orig_qweight
                    converted_state_dict[qweight_key] = padded_qweight
                
                # Pad proj_up if it exists (shape: [out_features, rank])
                proj_up_key = f"{n}.proj_up"
                if proj_up_key in converted_state_dict:
                    orig_proj_up = converted_state_dict[proj_up_key]
                    padded_shape = list(orig_proj_up.shape)
                    padded_shape[0] = m.out_features_padded
                    padded_proj_up = torch.zeros(padded_shape, dtype=orig_proj_up.dtype, device=orig_proj_up.device)
                    padded_proj_up[:orig_proj_up.shape[0]] = orig_proj_up
                    converted_state_dict[proj_up_key] = padded_proj_up
                
                # Pad wscales if it exists (shape: [in_features // group_size, out_features])
                wscales_key = f"{n}.wscales"
                if wscales_key in converted_state_dict:
                    orig_wscales = converted_state_dict[wscales_key]
                    padded_shape = list(orig_wscales.shape)
                    padded_shape[1] = m.out_features_padded
                    padded_wscales = torch.ones(padded_shape, dtype=orig_wscales.dtype, device=orig_wscales.device)
                    padded_wscales[:, :orig_wscales.shape[1]] = orig_wscales
                    converted_state_dict[wscales_key] = padded_wscales
                
                # Pad wcscales if it exists (shape: [out_features,])
                wcscales_key = f"{n}.wcscales"
                if wcscales_key in converted_state_dict:
                    orig_wcscales = converted_state_dict[wcscales_key]
                    padded_wcscales = torch.ones(m.out_features_padded, dtype=orig_wcscales.dtype, device=orig_wcscales.device)
                    padded_wcscales[:orig_wcscales.shape[0]] = orig_wcscales
                    converted_state_dict[wcscales_key] = padded_wcscales
        
        # Load the weights with strict verification
        missing_keys, unexpected_keys = transformer.load_state_dict(
            converted_state_dict, strict=False
        )
        
        # Filter out expected missing keys:
        # - wcscales/wtscale for unquantized layers
        # - norm2 weights when cross_attn_norm is False (norm2 is nn.Identity)
        # - blocks.*.scale_shift_table as these are randomly initialized per block
        real_missing = [k for k in missing_keys 
                       if '.wcscales' not in k 
                       and '.wtscale' not in k
                       and '.norm2.' not in k
                       and not (k.startswith('blocks.') and k.endswith('.scale_shift_table'))]
        if real_missing:
            print(f"CRITICAL: Missing keys: {real_missing[:10]}")
            if len(real_missing) > 10:
                print(f"... and {len(real_missing) - 10} more")
            raise RuntimeError(f"Failed to load {len(real_missing)} required keys!")
        
        if unexpected_keys:
            print(f"WARNING: Unexpected keys: {unexpected_keys[:10]}")
            if len(unexpected_keys) > 10:
                print(f"... and {len(unexpected_keys) - 10} more")
        
        print(f"Successfully loaded model with {len(converted_state_dict)} parameters")
        
        return transformer


def convert_wan_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Wan state dict from checkpoint format to the format expected by NunchakuWanTransformer3DModel.
    This maintains compatibility with the saved quantized weights.
    
    CRITICAL: The checkpoint uses fused QKV projections for self-attention.
    We keep them fused instead of splitting.
    
    NOTE: The global scale_shift_table in the checkpoint is for the main model only.
    Each block has its own scale_shift_table that is initialized randomly.
    """
    new_state_dict = {}
    
    # Process each key
    for k, v in state_dict.items():
        # Handle global scale_shift_table - it stays at the top level for the main model
        if k == "scale_shift_table":
            new_state_dict[k] = v
            continue
        # Handle fused QKV weights for self-attention - keep them fused!
        if "blocks." in k and ".qkv_proj." in k and "context" not in k:
            # Extract block number and weight type
            block_num = k.split(".")[1]
            weight_type = k.split(".qkv_proj.")[-1]
            
            # Rename weight types if needed
            renamed_weight_type = weight_type
            if weight_type == "lora_up":
                renamed_weight_type = "proj_up"
            elif weight_type == "lora_down":
                renamed_weight_type = "proj_down"
            elif weight_type == "smooth":
                renamed_weight_type = "smooth_factor"
            elif weight_type == "smooth_orig":
                renamed_weight_type = "smooth_factor_orig"
            
            # Keep as fused QKV for self-attention
            new_state_dict[f"blocks.{block_num}.attn1.to_qkv.{renamed_weight_type}"] = v
            continue
        
        # Normal processing for non-fused weights
        new_k = k
        
        # Map quantized weight names for transformer blocks
        if "blocks." in k:
            # Skip qkv_proj as we already handled it
            if ".qkv_proj." in k and "context" not in k:
                continue  # Already handled above
                
            elif ".q_proj." in k and "context" not in k:
                new_k = k.replace(".q_proj.", ".attn1.to_q.")
            elif ".k_proj." in k and "context" not in k and "add_" not in k:
                new_k = k.replace(".k_proj.", ".attn1.to_k.")
            elif ".v_proj." in k and "context" not in k and "add_" not in k:
                new_k = k.replace(".v_proj.", ".attn1.to_v.")
            
            # Cross-attention projections - CRITICAL FIX
            # Handle indexed projections qkv_proj_context_0/1/2 from checkpoint
            # These are the ACTUAL keys in the checkpoint!
            elif ".qkv_proj_context_0." in k:
                # Q projection (quantized)
                new_k = k.replace(".qkv_proj_context_0.", ".attn2.to_q.")
                # Rename lora/smooth parameters for quantized layers
                if "lora_down" in new_k:
                    new_k = new_k.replace("lora_down", "proj_down")
                elif "lora_up" in new_k:
                    new_k = new_k.replace("lora_up", "proj_up")
                elif "smooth" in new_k and not "smooth_" in new_k:
                    new_k = new_k.replace("smooth", "smooth_factor")
                elif "smooth_orig" in new_k:
                    new_k = new_k.replace("smooth_orig", "smooth_factor_orig")
            elif ".qkv_proj_context_1." in k:
                # K projection (NOT quantized - regular nn.Linear)
                new_k = k.replace(".qkv_proj_context_1.", ".attn2.to_k.")
            elif ".qkv_proj_context_2." in k:
                # V projection (NOT quantized - regular nn.Linear)
                new_k = k.replace(".qkv_proj_context_2.", ".attn2.to_v.")
            # Other cross-attention patterns (shouldn't exist with indexed, but keep for safety)
            elif ".qkv_proj_context." in k or ".q_proj_context." in k:
                new_k = k.replace(".qkv_proj_context.", ".attn2.to_q.")
                new_k = new_k.replace(".q_proj_context.", ".attn2.to_q.")
            elif ".k_proj_context." in k:
                new_k = k.replace(".k_proj_context.", ".attn2.to_k.")
            elif ".v_proj_context." in k:
                new_k = k.replace(".v_proj_context.", ".attn2.to_v.")
            
            # Additional KV projections (I2V)
            elif ".add_kv_proj." in k:
                new_k = k.replace(".add_kv_proj.", ".attn2.to_added_kv.")
            elif ".add_k_proj." in k:
                new_k = k.replace(".add_k_proj.", ".attn2.add_k_proj.")
            elif ".add_v_proj." in k:
                new_k = k.replace(".add_v_proj.", ".attn2.add_v_proj.")
            
            # Output projections
            elif ".out_proj_context." in k:
                new_k = k.replace(".out_proj_context.", ".attn2.to_out.0.")
            elif ".out_proj." in k and "context" not in k:
                new_k = k.replace(".out_proj.", ".attn1.to_out.0.")
            
            # FFN projections - mlp_fc1/mlp_fc2 need to map to BOTH locations
            # fc1 and net[0].proj are the same object, so we need both keys in state dict
            # fc2 and net[2] are the same object, so we need both keys in state dict
            elif ".mlp_fc1." in k:
                # Map to both fc1 and net.0.proj (they're the same tensor)
                fc1_key = k.replace(".mlp_fc1.", ".ffn.fc1.")
                net_key = k.replace(".mlp_fc1.", ".ffn.net.0.proj.")
                
                # Rename lora/smooth parameters
                for key in [fc1_key, net_key]:
                    if "lora_down" in key:
                        key = key.replace("lora_down", "proj_down")
                    elif "lora_up" in key:
                        key = key.replace("lora_up", "proj_up")
                    elif "smooth" in key and not "smooth_" in key:
                        key = key.replace("smooth", "smooth_factor")
                    elif "smooth_orig" in key:
                        key = key.replace("smooth_orig", "smooth_factor_orig")
                    new_state_dict[key] = v
                continue  # Skip normal processing since we handled both
            elif ".mlp_fc2." in k:
                # Map to both fc2 and net.2 (they're the same tensor)
                fc2_key = k.replace(".mlp_fc2.", ".ffn.fc2.")
                net_key = k.replace(".mlp_fc2.", ".ffn.net.2.")
                
                # Rename lora/smooth parameters
                for key in [fc2_key, net_key]:
                    if "lora_down" in key:
                        key = key.replace("lora_down", "proj_down")
                    elif "lora_up" in key:
                        key = key.replace("lora_up", "proj_up")
                    elif "smooth" in key and not "smooth_" in key:
                        key = key.replace("smooth", "smooth_factor")
                    elif "smooth_orig" in key:
                        key = key.replace("smooth_orig", "smooth_factor_orig")
                    new_state_dict[key] = v
                continue  # Skip normal processing since we handled both
            
            # Normalization layers
            elif ".norm_q." in k and "context" not in k:
                new_k = k.replace(".norm_q.", ".attn1.norm_q.")
            elif ".norm_k." in k and "context" not in k and "added" not in k:
                new_k = k.replace(".norm_k.", ".attn1.norm_k.")
            elif ".norm_q_context." in k:
                new_k = k.replace(".norm_q_context.", ".attn2.norm_q.")
            elif ".norm_k_context." in k:
                new_k = k.replace(".norm_k_context.", ".attn2.norm_k.")
            elif ".norm_added_k." in k:
                new_k = k.replace(".norm_added_k.", ".attn2.norm_added_k.")
        
        # Rename lora_down/lora_up to proj_down/proj_up
        if ".lora_down" in new_k:
            new_k = new_k.replace(".lora_down", ".proj_down")
        if ".lora_up" in new_k:
            new_k = new_k.replace(".lora_up", ".proj_up")
        
        # Rename smooth/smooth_orig to smooth_factor/smooth_factor_orig
        if new_k.endswith(".smooth"):
            new_k = new_k.replace(".smooth", ".smooth_factor")
        if new_k.endswith(".smooth_orig"):
            new_k = new_k.replace(".smooth_orig", ".smooth_factor_orig")
        
        new_state_dict[new_k] = v
    
    return new_state_dict
