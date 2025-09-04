"""
Merge split safetensors model files (deprecated format) into a single safetensors file.

**Example usage**

.. code-block:: bash

    python -m nunchaku.merge_safetensors -i <input_path_or_repo> -o <output_path>

**Arguments**

- ``-i``, ``--input-path`` (Path): Path to the model directory or HuggingFace repo.
- ``-o``, ``--output-path`` (Path): Path to save the merged safetensors file.

It will combine the ``unquantized_layers.safetensors`` and ``transformer_blocks.safetensors``
files (and associated config files) from a local directory or a HuggingFace Hub repository
into a single safetensors file with appropriate metadata.

**Main Function**

:func:`merge_safetensors`
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import save_file

from .utils import load_state_dict_in_safetensors


def detect_model_type(config_content: str) -> str:
    """
    Detect the model type from the config content.
    
    Parameters
    ----------
    config_content : str
        The content of the config.json file.
        
    Returns
    -------
    str
        The detected model type: 'flux', 'sana', or 'wan'.
    """
    try:
        config = json.loads(config_content)
        # Check for model-specific indicators in the config
        if 'joint_attention_dim' in config or 'pooled_projection_dim' in config:
            return 'flux'
        elif 'sample_size' in config and isinstance(config.get('sample_size'), list):
            # WAN models typically have sample_size as a list [frames, height, width]
            return 'wan'
        elif 'linear_head_dim' in config:
            return 'sana'
        else:
            # Default fallback based on common patterns
            if config.get('num_attention_heads', 0) == 24:
                return 'flux'
            elif config.get('num_attention_heads', 0) == 36:
                return 'sana'
            else:
                return 'wan'
    except:
        # If we can't parse the config, default to flux for backward compatibility
        return 'flux'


def get_model_class_name(model_type: str) -> str:
    """
    Get the model class name based on the model type.
    
    Parameters
    ----------
    model_type : str
        The model type: 'flux', 'sana', or 'wan'.
        
    Returns
    -------
    str
        The corresponding model class name.
    """
    model_class_map = {
        'flux': 'NunchakuFluxTransformer2dModel',
        'sana': 'NunchakuSanaTransformer2DModel', 
        'wan': 'NunchakuWanTransformer3DModel'
    }
    return model_class_map.get(model_type.lower(), 'NunchakuFluxTransformer2dModel')


def merge_safetensors(
    pretrained_model_name_or_path: str | os.PathLike[str], 
    model_type: str = None,
    **kwargs
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Merge split safetensors model files into a single state dict and metadata.

    This function loads the ``unquantized_layers.safetensors`` and ``transformer_blocks.safetensors``
    files (and associated config files) from a local directory or a HuggingFace Hub repository,
    and merges them into a single state dict and metadata dictionary.

    Parameters
    ----------
    pretrained_model_name_or_path : str or os.PathLike
        Path to the model directory or HuggingFace repo.
    model_type : str, optional
        The model type: 'flux', 'sana', or 'wan'. If not provided, will be auto-detected.
    **kwargs
        Additional keyword arguments for subfolder, comfy_config_path, and HuggingFace download options.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, str]]
        The merged state dict and metadata dictionary.

        - **state_dict**: The merged model state dict.
        - **metadata**: Dictionary containing ``config``, ``comfy_config``, ``model_class``, and ``quantization_config``.
    """
    subfolder = kwargs.get("subfolder", None)
    comfy_config_path = kwargs.get("comfy_config_path", None)

    if isinstance(pretrained_model_name_or_path, str):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    if pretrained_model_name_or_path.exists():
        dirpath = pretrained_model_name_or_path if subfolder is None else pretrained_model_name_or_path / subfolder
        unquantized_part_path = dirpath / "unquantized_layers.safetensors"
        transformer_block_path = dirpath / "transformer_blocks.safetensors"
        config_path = dirpath / "config.json"
        if comfy_config_path is None:
            comfy_config_path = dirpath / "comfy_config.json"
    else:
        download_kwargs = {
            "subfolder": subfolder,
            "repo_type": "model",
            "revision": kwargs.get("revision", None),
            "cache_dir": kwargs.get("cache_dir", None),
            "local_dir": kwargs.get("local_dir", None),
            "user_agent": kwargs.get("user_agent", None),
            "force_download": kwargs.get("force_download", False),
            "proxies": kwargs.get("proxies", None),
            "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
            "token": kwargs.get("token", None),
            "local_files_only": kwargs.get("local_files_only", None),
            "headers": kwargs.get("headers", None),
            "endpoint": kwargs.get("endpoint", None),
            "resume_download": kwargs.get("resume_download", None),
            "force_filename": kwargs.get("force_filename", None),
            "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
        }
        unquantized_part_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="unquantized_layers.safetensors", **download_kwargs
        )
        transformer_block_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="transformer_blocks.safetensors", **download_kwargs
        )
        config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="config.json", **download_kwargs
        )
        comfy_config_path = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path), filename="comfy_config.json", **download_kwargs
        )

    unquantized_part_sd = load_state_dict_in_safetensors(unquantized_part_path)
    transformer_block_sd = load_state_dict_in_safetensors(transformer_block_path)
    state_dict = unquantized_part_sd
    state_dict.update(transformer_block_sd)

    # Read config to detect model type if not specified
    config_content = Path(config_path).read_text()
    if model_type is None:
        model_type = detect_model_type(config_content)
        print(f"Auto-detected model type: {model_type}")
    else:
        print(f"Using specified model type: {model_type}")

    # Determine precision
    precision = "int4"
    for v in state_dict.values():
        assert isinstance(v, torch.Tensor)
        if v.dtype in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e8m0fnu,
        ]:
            precision = "fp4"
    
    quantization_config = {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "scale_dtype": "fp8_e4m3_nan" if precision == "fp4" else None,
            "group_size": 16 if precision == "fp4" else 64,
        },
    }
    
    # Get the appropriate model class name
    model_class_name = get_model_class_name(model_type)
    
    # Check if comfy_config exists for the model (not all models may have it)
    comfy_config_content = ""
    if Path(comfy_config_path).exists():
        comfy_config_content = Path(comfy_config_path).read_text()
    
    metadata = {
        "config": config_content,
        "model_class": model_class_name,
        "quantization_config": json.dumps(quantization_config),
    }
    
    # Only add comfy_config if it exists
    if comfy_config_content:
        metadata["comfy_config"] = comfy_config_content
    
    return state_dict, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge split safetensors model files into a single safetensors file."
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to model directory. It can also be a huggingface repo.",
    )
    parser.add_argument("-o", "--output-path", type=Path, required=True, help="Path to output path")
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        choices=["flux", "sana", "wan"],
        default=None,
        help="Model type: flux, sana, or wan. If not specified, will be auto-detected from config.",
    )
    args = parser.parse_args()
    
    state_dict, metadata = merge_safetensors(args.input_path, model_type=args.model_type)
    output_path = Path(args.output_path)
    dirpath = output_path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path, metadata=metadata)
    print(f"Successfully merged model to: {output_path}")
    print(f"Model class: {metadata['model_class']}")
