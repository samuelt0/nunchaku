import torch
from diffusers import WanPipeline, WanTransformer3DModel

path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
dtype = torch.bfloat16
transformer = WanTransformer3DModel.from_pretrained(path, torch_dtype=dtype, subfolder="transformer")
transformer_2 = WanTransformer3DModel.from_pretrained(path, torch_dtype=dtype, subfolder="transformer_2")
pipeline = WanPipeline.from_pretrained(
    path, transformer=transformer, transformer_2=transformer_2, torch_dtype=dtype)
prompt = "A cat walks on the grass, realistic"

# Also need to import export_to_video
from diffusers.utils import export_to_video

output = pipeline(
    prompt=prompt,
    height=360,
    width=640,
    num_frames=30,
    guidance_scale=4,
    guidance_scale_2=3,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
