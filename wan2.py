import torch
from diffusers import WanPipeline

from nunchaku import NunchakuWanTransformer3DModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
print(precision)
# Using the merged safetensors files
transformer = NunchakuWanTransformer3DModel.from_pretrained(
    "/data/samuel/nunchaku/int4-high/merged.safetensors", offload=False, precision=precision
)  # set offload to False if you want to disable offloading
transformer_2 = NunchakuWanTransformer3DModel.from_pretrained(
    "/data/samuel/nunchaku/int4-low/merged.safetensors", offload=False, precision=precision
)  # set offload to False if you want to disable offloading
pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", transformer=transformer, transformer_2=transformer_2, torch_dtype=torch.bfloat16
)  # no need to set the device here
pipeline.enable_sequential_cpu_offload()  # diffusers' offloading
prompt = "A cat walks on the grass, realistic"

# Also need to import export_to_video
from diffusers.utils import export_to_video

output = pipeline(
    prompt=prompt,
    height=160,
    width=160,
    num_frames=29,
    guidance_scale=4,
    guidance_scale_2=3,
    num_inference_steps=20
).frames[0]
export_to_video(output, "output.mp4", fps=15)
