import os
from typing import Optional

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import utils
from safetensors.torch import load_file
from torch import nn

from .utils import NunchakuModelLoaderMixin
from ...utils import get_precision
from ..._C import QuantizedWanModel, utils as cutils



class NunchakuWanTransformerBlocks(nn.Module):
    def __init__(self, m: QuantizedWanModel, dtype: torch.dtype, device: str | torch.device):
        super().__init__()
        self.m = m
        self.dtype = dtype        
        self.device = device

    def forward(
        self,
        hidden_states: torch.Tensor,           
        encoder_hidden_states: torch.Tensor,   
        timestep_proj: torch.Tensor,           
        rotary_emb: torch.Tensor,              
        skip_first_layer: Optional[bool] = False,
    ) -> torch.Tensor:

        batch_size = hidden_states.shape[0]
        original_dtype = hidden_states.dtype
        original_device  = hidden_states.device

        hidden_states_t = hidden_states.to(self.dtype).to(self.device)
        enc_states_t = encoder_hidden_states.to(self.dtype).to(self.device)
        temb_t = timestep_proj.to(self.dtype).to(self.device)

        result = self.m.forward(
            hidden_states_t,   
            temb_t,                     
            enc_states_t,              
            torch.Tensor().to(self.device),  
            torch.Tensor().to(self.device),
        )

        return result.to(original_dtype).to(original_device)

    def forward_layer_at(
        self,
        idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype   = hidden_states.dtype
        original_device  = hidden_states.device

        hidden_states_t  = hidden_states.to(self.dtype).to(self.device)
        enc_states_t     = encoder_hidden_states.to(self.dtype).to(self.device)
        temb_t           = timestep_proj.to(self.dtype).to(self.device)

        out = self.m.forward_layer( 
            idx,
            hidden_states_t,
            temb_t,
            enc_states_t,
            torch.Tensor().to(self.device),
        )

        return out.to(original_dtype).to(original_device)

    def __del__(self):
        self.m.reset()


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        device = kwargs.get("device", "cuda")
        pag_layers = kwargs.get("pag_layers", [])
        precision = get_precision(kwargs.get("precision", "auto"),
                                    device,
                                    pretrained_model_name_or_path)

        transformer, unquantized_part_path, transformer_block_path = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )

        use_fp4 = precision == "fp4"
        quant_mod = load_quantized_module(
            transformer,
            transformer_block_path,
            device=device,
            use_fp4=use_fp4,
        )

        transformer.inject_quantized_module(quant_mod, device)

        unquantized_sd = load_file(unquantized_part_path)
        transformer.load_state_dict(unquantized_sd, strict=False)

        return transformer
        
    def inject_quantized_module(self, m: QuantizedWanModel, device: str | torch.device = "cuda"):
        self.blocks = torch.nn.ModuleList([NunchakuWanTransformerBlocks(m, self.dtype, device)])
        return self


def load_quantized_module(
    net: WanTransformer3DModel,
    path: str,
    device: str | torch.device = "cuda",
    use_fp4: bool = False,
) -> QuantizedWanModel:
    device = torch.device(device)
    assert device.type == "cuda"

    m = QuantizedWanModel()
    cutils.disable_memory_auto_release()
    m.init(net.config, use_fp4, net.dtype == torch.bfloat16, 0 if device.index is None else device.index)
    m.load(path)
    return m

def inject_quantized_module(
    net: WanTransformer3DModel, m: QuantizedWanModel, device: torch.device
) -> WanTransformer3DModel:
    net.blocks = torch.nn.ModuleList([NunchakuWanTransformerBlocks(m, net.dtype, device)])
    return net
