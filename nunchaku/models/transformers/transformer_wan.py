import os
from typing import Optional, Union

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import utils
from safetensors.torch import load_file
from torch import nn

from .utils import NunchakuModelLoaderMixin
from ...utils import get_precision
from ..._C import QuantizedWanModel, utils as cutils


def load_quantized_module(
    path: str,
    cfg: dict,
    device: Union[str, torch.device] = "cuda",
    bf16: bool = True,
) -> QuantizedWanModel:
    device = torch.device(device)
    assert device.type == "cuda"
    m = QuantizedWanModel()
    cutils.disable_memory_auto_release()
    m.init(cfg, bf16, 0 if device.index is None else device.index)
    m.load(path)
    return m


class NunchakuWanTransformer3DModel(WanTransformer3DModel, NunchakuModelLoaderMixin):
    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        device = kwargs.get("device", "cuda")
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        precision = get_precision(kwargs.get("precision", "auto"), device, pretrained_model_name_or_path)

        transformer, unquantized_path, quantized_path = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )

        quant_mod = load_quantized_module(
            quantized_path,
            transformer.config,
            device=device,
            bf16=torch_dtype == torch.bfloat16,
        )
        transformer.inject_quantized_module(quant_mod, device)
        transformer.to_empty(device=device)

        unquant_sd = load_file(unquantized_path)
        transformer.load_state_dict(unquant_sd, strict=False)
        return transformer

    def inject_quantized_module(self, m: QuantizedWanModel, device: Union[str, torch.device] = "cuda"):
        self.quant_module = m
        self.device_for_quant = device
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        orig_dtype, orig_device = hidden_states.dtype, hidden_states.device

        hs = hidden_states.to(self.dtype).to(self.device_for_quant)
        ts = timestep.to(self.dtype).to(self.device_for_quant)
        txt = encoder_hidden_states.to(self.dtype).to(self.device_for_quant)
        img = (
            encoder_hidden_states_image.to(self.dtype).to(self.device_for_quant)
            if encoder_hidden_states_image is not None
            else None
        )

        out = self.quant_module.forward(hs, ts, txt, img).to(orig_dtype).to(orig_device)

        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=out)
