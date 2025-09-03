"""
Example script for running Wan 2.2 models with Nunchaku quantization.

This script demonstrates how to load and use a quantized Wan model for video generation.
"""

import torch
from diffusers import DiffusionPipeline

from nunchaku import NunchakuWanTransformer3DModel
from nunchaku.utils import get_precision


def main():
    # Auto-detect precision based on GPU capabilities
    precision = get_precision()  # auto-detect 'int4' or 'fp4' based on your GPU
    
    # Path to your quantized Wan model
    # This should be the output directory from convert.py
    quantized_model_path = "path/to/your/quantized-wan-model"
    
    # Alternative: Load from a single safetensors file
    # quantized_model_path = "path/to/your/svdq-int4_r32-wan2.2.safetensors"
    
    # Load the quantized transformer
    transformer = NunchakuWanTransformer3DModel.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
        precision=precision,
        offload=False  # Set to True if you want to enable offloading
    )
    
    # Create the pipeline with the quantized transformer
    # Note: You'll need to specify the correct Wan pipeline class
    # This is a placeholder - replace with actual Wan pipeline when available
    pipeline = DiffusionPipeline.from_pretrained(
        "path/to/wan-2.2-base",  # Base model path
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    # Move pipeline to CUDA
    pipeline = pipeline.to("cuda")
    
    # Optional: Enable sequential CPU offloading for memory efficiency
    # pipeline.enable_sequential_cpu_offload()
    
    # Generate a video
    prompt = "A beautiful sunset over the ocean, waves gently crashing on the shore"
    
    # Video generation parameters
    video = pipeline(
        prompt=prompt,
        num_frames=16,  # Number of frames to generate
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).frames[0]
    
    # Save the generated video
    # Note: You'll need to implement video saving logic based on your pipeline output
    print(f"Generated video with {len(video)} frames using {precision} precision")
    
    # For image-to-video (I2V) models:
    # from PIL import Image
    # init_image = Image.open("path/to/init/image.jpg")
    # video = pipeline(
    #     prompt=prompt,
    #     image=init_image,
    #     num_frames=16,
    #     num_inference_steps=50,
    #     guidance_scale=7.5,
    # ).frames[0]


def convert_and_run():
    """
    Example showing the full workflow from conversion to inference.
    """
    import subprocess
    import sys
    
    # Step 1: Convert the model using convert.py
    print("Converting Wan model to Nunchaku format...")
    
    # Path to your DeepCompressor quantized checkpoint
    quant_path = "path/to/deepcompressor/checkpoint"
    output_path = "path/to/output/nunchaku-wan-model"
    
    # Run the conversion
    cmd = [
        sys.executable, "convert.py",
        "--quant-path", quant_path,
        "--output-root", output_path,
        "--model-name", "wan-2.2"
    ]
    
    # Add --float-point flag if you want FP4 quantization
    # cmd.append("--float-point")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Model converted successfully to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return
    
    # Step 2: Load and use the converted model
    print("\nLoading quantized model...")
    
    precision = get_precision()
    transformer = NunchakuWanTransformer3DModel.from_pretrained(
        f"{output_path}/wan-2.2",
        torch_dtype=torch.bfloat16,
        device="cuda",
        precision=precision
    )
    
    print(f"Model loaded successfully with {precision} precision")
    print("Ready for video generation!")


if __name__ == "__main__":
    # Run the basic example
    main()
    
    # Or run the full conversion + inference workflow
    # convert_and_run()
