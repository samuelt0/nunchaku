import torch
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from torch import nn

from ..ops.fused import fused_gelu_mlp
from .linear import SVDQW4A4Linear


class NunchakuBaseAttention(nn.Module):
    def __init__(self, processor: str = "flashattn2", *args, **kwargs):
        super(NunchakuBaseAttention, self).__init__()
        self.processor = None
        self.set_processor(processor)

    def set_processor(self, processor: str):
        raise NotImplementedError("Subclass must implement this method")


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuFeedForward(FeedForward):
    def __init__(self, ff: FeedForward, **kwargs):
        super(FeedForward, self).__init__()
        # Copy the network structure
        self.net = nn.ModuleList()
        
        # Initialize fc1 and fc2
        self.fc1 = None
        self.fc2 = None
        
        # Process each layer in the original network
        for i, module in enumerate(ff.net):
            if isinstance(module, GELU) and hasattr(module, 'proj'):
                # GELU with embedded linear projection (WAN structure)
                # Keep the original GELU, but replace its proj with quantized version
                module.proj = SVDQW4A4Linear.from_linear(module.proj, **kwargs)
                self.fc1 = module.proj
                self.net.append(module)
            elif isinstance(module, nn.Linear):
                # Standalone linear layer - quantize it
                quantized = SVDQW4A4Linear.from_linear(module, **kwargs)
                self.net.append(quantized)
                # The second linear layer we encounter is fc2
                if self.fc1 is not None and self.fc2 is None:
                    self.fc2 = quantized
                elif self.fc1 is None:
                    # If we haven't seen fc1 yet, this must be it
                    self.fc1 = quantized
            else:
                # Keep other layers as-is (e.g., Dropout)
                self.net.append(module)
        
        # Set act_unsigned for fc2 if it exists
        if self.fc2 is not None:
            self.fc2.act_unsigned = self.fc2.precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # WAN has GELU with embedded linear (fc1 inside GELU.proj), 
        # which is incompatible with fused_gelu_mlp that expects separate layers.
        # The fused operation would try to call fc1.quantize directly on input,
        # missing the GELU activation. So we use sequential execution.
        
        # Check if fc1 is embedded in GELU (WAN structure)
        if (isinstance(self.net[0], GELU) and hasattr(self.net[0], 'proj') and 
            self.net[0].proj == self.fc1):
            # WAN structure: fc1 is inside GELU, must use sequential execution
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states
        
        # Check if we have a standard structure suitable for fused operation
        # (e.g., separate linear layer followed by GELU, not GELU with embedded linear)
        if (self.fc1 is not None and self.fc2 is not None and 
            not isinstance(self.net[0], GELU)):
            # Standard structure with separate layers - can use fused operation
            return fused_gelu_mlp(hidden_states, self.fc1, self.fc2)
        
        # Fallback to sequential execution for any other structure
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
