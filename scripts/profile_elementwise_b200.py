"""Profile Nunchaku element-wise kernels on B200 with FP4 GEMM stubbed.

This script enables profiling of element-wise kernels (activations, layernorms, etc.)
on B200 GPUs where FP4 GEMM kernels don't work (requires SM 120+). The GEMM kernels
are stubbed out to allow the rest of the pipeline to execute.

Usage:
    # Basic run (outputs will be garbage due to stubbed GEMM, but kernel timings are valid)
    python scripts/profile_elementwise_b200.py

    # Profile with Nsight Systems
    nsys profile -c cudaProfilerApi python scripts/profile_elementwise_b200.py

    # Profile with specific options
    nsys profile -c cudaProfilerApi -o profile_elementwise --stats=true python scripts/profile_elementwise_b200.py


    nsys profile -o nunchaku-elm python3 scripts/profile_elementwise_b200.py
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku._C import utils as cutils


def main():
    parser = argparse.ArgumentParser(description="Profile Nunchaku element-wise kernels on B200")
    parser.add_argument(
        "--model",
        type=str,
        default="nunchaku-tech/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base Flux model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp4",
        choices=["fp4", "int4"],
        help="Quantization precision",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=1,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Profile only the transformer (recommended for focused profiling)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps for full pipeline profiling",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Precision: {args.precision}")

    # Enable stub mode for FP4 GEMM
    if args.precision == "fp4":
        cutils.set_stub_fp4_gemm(True)
        print("FP4 GEMM stub mode enabled")

    # Load transformer
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        args.model, precision=args.precision
    )

    # Load full pipeline
    from diffusers import FluxPipeline

    pipeline = FluxPipeline.from_pretrained(
        args.base_model, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    if args.transformer_only:
        # Profile transformer only
        print("Capturing transformer inputs...")
        inputs = {}

        def capture_hook(module, args, kwargs):
            inputs["args"] = args
            inputs["kwargs"] = kwargs

        pipeline.transformer.register_forward_pre_hook(capture_hook, with_kwargs=True)

        # Run once to capture inputs
        with torch.no_grad():
            pipeline(
                "test",
                num_inference_steps=1,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                output_type="latent",
            )

        print(f"Warming up transformer ({args.warmup_steps} iterations)...")
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                pipeline.transformer(*inputs["args"], **inputs["kwargs"])
        torch.cuda.synchronize()

        print(f"Profiling transformer ({args.profile_steps} iterations)...")
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("transformer_inference")

        for i in range(args.profile_steps):
            nvtx.range_push(f"iteration_{i}")
            with torch.no_grad():
                pipeline.transformer(*inputs["args"], **inputs["kwargs"])
            nvtx.range_pop()

        torch.cuda.synchronize()
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    else:
        # Profile full pipeline
        print(f"Warming up pipeline ({args.warmup_steps} iterations)...")
        for _ in range(args.warmup_steps):
            with torch.no_grad():
                pipeline(
                    "test",
                    num_inference_steps=1,
                    output_type="latent",
                )
        torch.cuda.synchronize()

        print(f"Profiling full pipeline ({args.profile_steps} iterations)...")
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("flux_inference")

        for i in range(args.profile_steps):
            nvtx.range_push(f"iteration_{i}")
            with torch.no_grad():
                pipeline(
                    "A cat sitting on a windowsill",
                    num_inference_steps=args.num_inference_steps,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    output_type="latent",
                )
            nvtx.range_pop()

        torch.cuda.synchronize()
        nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    print("Profiling complete!")
    print("\nTo analyze results, run with Nsight Systems:")
    print("  nsys profile -c cudaProfilerApi python scripts/profile_elementwise_b200.py")
    print("\nNote: Output images will be garbage due to stubbed GEMM, but kernel timings are valid.")


if __name__ == "__main__":
    main()
