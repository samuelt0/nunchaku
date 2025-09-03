import torch
from diffusers import WanPipeline

from nunchaku import NunchakuWanTransformer3DModel
from nunchaku.utils import get_precision

precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
transformer = NunchakuWanTransformer3DModel.from_pretrained(
    "/FirstIntelligence/samuel/nunchaku/wan2-high", offload=True
)  # set offload to False if you want to disable offloading
transformer_2 = NunchakuWanTransformer3DModel.from_pretrained(
    "/FirstIntelligence/samuel/nunchaku/wan2-high", offload=True
)  # set offload to False if you want to disable offloading
pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", transformer=transformer, transformer_2=transformer_2, torch_dtype=torch.bfloat16
)  # no need to set the device here
pipeline.enable_sequential_cpu_offload()  # diffusers' offloading
prompt = "A cat walks on the grass, realistic"

output = pipe(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=4,
    guidance_scale_2=3,
).frames[0]
export_to_video(output, "output.mp4", fps=15)