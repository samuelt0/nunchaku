"""
SVDQW4A4Linear with automatic padding for GEMM kernel alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import SVDQW4A4Linear


class SVDQW4A4LinearPadded(SVDQW4A4Linear):
    """
    SVDQW4A4Linear with automatic padding to ensure output dimensions are divisible by BLOCK_N.
    This wrapper handles dimensions that don't align with GEMM kernel requirements.
    """
    
    BLOCK_N = 128  # GEMM kernel block size requirement
    
    def __init__(self, in_features, out_features, bias=True, device=None, torch_dtype=None, **kwargs):
        # Calculate padded output dimension
        self.original_out_features = out_features
        self.out_features_padded = ((out_features + self.BLOCK_N - 1) // self.BLOCK_N) * self.BLOCK_N
        self.needs_padding = self.out_features_padded != out_features
        
        # Initialize with padded dimensions - note we pass torch_dtype not dtype
        if torch_dtype is not None:
            kwargs['torch_dtype'] = torch_dtype
        super().__init__(in_features, self.out_features_padded, bias=bias, device=device, **kwargs)
        
        # Store the actual output features we want
        self.true_out_features = out_features
        
        if self.needs_padding:
            print(f"SVDQW4A4LinearPadded: Padding output from {out_features} to {self.out_features_padded}")
    
    def forward(self, x):
        """Forward pass with automatic padding and unpadding."""
        # Run the forward pass with padded dimensions
        output = super().forward(x)
        
        # Remove padding from output if needed
        if self.needs_padding:
            output = output[..., :self.original_out_features]
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """Create padded quantized linear from regular linear layer."""
        # First create a regular quantized version to get all the quantized attributes
        temp_quantized = SVDQW4A4Linear.from_linear(linear, **kwargs)
        
        # Create new padded instance
        padded = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            torch_dtype=linear.weight.dtype,
            **kwargs
        )
        
        # Copy all the quantized weights and parameters from temp_quantized
        # These are the key attributes from SVDQW4A4Linear
        for attr in ['qweight', 'wscales', 'smooth_factor', 'smooth_factor_orig', 
                     'proj_down', 'proj_up', 'wcscales', 'wtscale']:
            if hasattr(temp_quantized, attr):
                src_value = getattr(temp_quantized, attr)
                if src_value is not None:
                    if isinstance(src_value, nn.Parameter):
                        # For parameters, we need to handle padding for output-related tensors
                        if padded.needs_padding and attr in ['qweight', 'proj_up']:
                            # qweight shape: (out_features, in_features // 2)
                            # proj_up shape: (out_features, rank)
                            # We need to pad the out_features dimension
                            padded_shape = list(src_value.shape)
                            padded_shape[0] = padded.out_features_padded
                            padded_tensor = torch.zeros(padded_shape, dtype=src_value.dtype, device=src_value.device)
                            padded_tensor[:src_value.shape[0]] = src_value
                            setattr(padded, attr, nn.Parameter(padded_tensor, requires_grad=src_value.requires_grad))
                        elif padded.needs_padding and attr == 'wscales':
                            # wscales shape: (in_features // group_size, out_features)
                            # We need to pad the out_features dimension (second dimension)
                            padded_shape = list(src_value.shape)
                            padded_shape[1] = padded.out_features_padded
                            padded_tensor = torch.ones(padded_shape, dtype=src_value.dtype, device=src_value.device)
                            padded_tensor[:, :src_value.shape[1]] = src_value
                            setattr(padded, attr, nn.Parameter(padded_tensor, requires_grad=src_value.requires_grad))
                        elif padded.needs_padding and attr == 'wcscales':
                            # wcscales shape: (out_features,)
                            padded_tensor = torch.ones(padded.out_features_padded, dtype=src_value.dtype, device=src_value.device)
                            padded_tensor[:src_value.shape[0]] = src_value
                            setattr(padded, attr, nn.Parameter(padded_tensor, requires_grad=src_value.requires_grad))
                        else:
                            # For other parameters or when no padding is needed, just copy
                            setattr(padded, attr, src_value)
                    else:
                        # For non-parameter attributes (like wtscale which is a float)
                        setattr(padded, attr, src_value)
        
        # Handle bias specially
        if linear.bias is not None:
            if padded.needs_padding:
                # Pad the bias with zeros
                padded_bias = torch.zeros(padded.out_features_padded, dtype=linear.bias.dtype, device=linear.bias.device)
                padded_bias[:linear.out_features] = linear.bias
                padded.bias = nn.Parameter(padded_bias)
            else:
                padded.bias = nn.Parameter(linear.bias.clone())
        
        return padded
