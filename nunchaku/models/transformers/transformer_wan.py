from typing import Any, Dict, Optional, Union

import logging
import os

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from huggingface_hub import utils
from packaging.version import Version
from safetensors.torch import load_file, save_file
from torch import nn

from .utils import NunchakuModelLoaderMixin, pad_tensor
from ..._C import QuantizedFluxModel, utils as cutils
from ...lora.flux.nunchaku_converter import fuse_vectors, to_nunchaku
from ...lora.flux.utils import is_nunchaku_format
from ...utils import get_precision, load_state_dict_in_safetensors

SVD_RANK = 32

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)