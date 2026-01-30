#!/usr/bin/env python3
"""
Memory Bandwidth Profiler for Nunchaku Element-wise Kernels.

Calculates memory bandwidth utilization (GB/s) for element-wise kernels in the
Flux transformer, comparing effective bandwidth to the GPU's theoretical peak.

Usage:
    # Step 1: Run with shape capture and layer timing
    python scripts/profile_bandwidth.py --model nunchaku-tech/nunchaku-flux.1-dev/...

    # Step 2: Run with nsys to get kernel timings
    nsys profile -o nunchaku_bw --export sqlite python scripts/profile_bandwidth.py

    # Step 3: Generate bandwidth report from nsys data
    python scripts/profile_bandwidth.py --nsys-db nunchaku_bw.sqlite --report

    # Shape sweep for different resolutions
    python scripts/profile_bandwidth.py --sweep-shapes --resolutions 512,1024,2048

    # B200 mode (stubs FP4 GEMM for element-wise profiling)
    python scripts/profile_bandwidth.py --b200-mode
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku._C import utils as cutils
from nunchaku.profiling import (
    ShapeCapture,
    TensorShape,
    calculate_bandwidth_from_ms,
    BandwidthResult,
    GPU_PEAK_BANDWIDTH,
    calculate_kernel_bandwidths,
    calculate_bandwidth_from_architecture,
    generate_kernel_bandwidth_report,
    FluxModelConfig,
    calculate_flux_kernel_bytes,
)
from nunchaku.profiling.shape_capture import (
    estimate_elementwise_bytes,
    get_flux_shapes_for_resolution,
)
from nunchaku.profiling.bandwidth_profiler import (
    LayerBandwidth,
    estimate_layer_elementwise_bytes,
    generate_bandwidth_report,
)


def capture_transformer_inputs(pipeline, prompt: str = "test", height: int = 1024, width: int = 1024):
    """Capture inputs to the transformer during a pipeline run.

    Args:
        pipeline: FluxPipeline instance
        prompt: Text prompt
        height: Image height
        width: Image width

    Returns:
        Dict with captured args and kwargs for transformer forward
    """
    inputs = {}

    def capture_hook(module, args, kwargs):
        inputs["args"] = args
        inputs["kwargs"] = kwargs

    handle = pipeline.transformer.register_forward_pre_hook(capture_hook, with_kwargs=True)

    with torch.no_grad():
        pipeline(
            prompt,
            num_inference_steps=1,
            height=height,
            width=width,
            guidance_scale=3.5,
            output_type="latent",
        )

    handle.remove()
    return inputs


def profile_transformer_layers(
    transformer: NunchakuFluxTransformer2dModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
    warmup_iters: int = 3,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> List[LayerBandwidth]:
    """Profile each transformer layer separately using forward_layer_at.

    Args:
        transformer: NunchakuFluxTransformer2dModel instance
        hidden_states: Image hidden states
        encoder_hidden_states: Text hidden states
        temb: Temporal embedding
        image_rotary_emb: Rotary embeddings
        num_double_blocks: Number of double transformer blocks
        num_single_blocks: Number of single transformer blocks
        warmup_iters: Number of warmup iterations per layer
        peak_gbs: Peak GPU bandwidth for efficiency calculation

    Returns:
        List of LayerBandwidth results for each layer
    """
    block = transformer.transformer_blocks[0]
    results = []

    # Get shapes for bandwidth estimation
    hidden_shape = tuple(hidden_states.shape)
    encoder_shape = tuple(encoder_hidden_states.shape)
    dtype_bytes = hidden_states.element_size()

    print(f"\nProfiling {num_double_blocks} double blocks + {num_single_blocks} single blocks...")
    print(f"Hidden states shape: {hidden_shape}")
    print(f"Encoder hidden states shape: {encoder_shape}")
    print(f"Dtype: {hidden_states.dtype} ({dtype_bytes} bytes/element)")
    print()

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Profile double blocks
    for layer_idx in range(num_double_blocks):
        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _, _ = block.forward_layer_at(
                    layer_idx,
                    hidden_states.clone(),
                    encoder_hidden_states.clone(),
                    temb,
                    image_rotary_emb,
                )
        torch.cuda.synchronize()

        # Timed run
        start_event.record()
        with torch.no_grad():
            nvtx.range_push(f"double_block_{layer_idx}")
            enc_out, hidden_out = block.forward_layer_at(
                layer_idx,
                hidden_states.clone(),
                encoder_hidden_states.clone(),
                temb,
                image_rotary_emb,
            )
            nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event)

        # Estimate element-wise bytes
        estimated_bytes = estimate_layer_elementwise_bytes(
            hidden_shape, encoder_shape, "double", dtype_bytes
        )

        # Calculate bandwidth
        bw_result = calculate_bandwidth_from_ms(estimated_bytes, time_ms, peak_gbs)

        layer_result = LayerBandwidth(
            layer_idx=layer_idx,
            layer_type="double",
            time_ms=time_ms,
            input_shape=hidden_shape,
            input_bytes=hidden_states.numel() * dtype_bytes,
            estimated_elementwise_bytes=estimated_bytes,
            bandwidth_result=bw_result,
        )
        results.append(layer_result)

        print(f"  {layer_result}")

    # Note: Single blocks are not individually callable via forward_layer_at
    # They are processed as part of the full forward pass
    # Here we just report the aggregate from the full forward

    return results


def profile_full_transformer(
    transformer: NunchakuFluxTransformer2dModel,
    inputs: Dict,
    warmup_iters: int = 3,
    profile_iters: int = 5,
    peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"],
) -> Tuple[float, ShapeCapture]:
    """Profile the full transformer forward pass.

    Args:
        transformer: NunchakuFluxTransformer2dModel instance
        inputs: Dict with args and kwargs from capture_transformer_inputs
        warmup_iters: Number of warmup iterations
        profile_iters: Number of profiling iterations
        peak_gbs: Peak GPU bandwidth

    Returns:
        Tuple of (average time in ms, ShapeCapture with recorded shapes)
    """
    capture = ShapeCapture()

    # Record input shapes
    hidden_states = inputs["kwargs"].get("hidden_states") or inputs["args"][0]
    encoder_hidden_states = inputs["kwargs"].get("encoder_hidden_states")
    pooled_projections = inputs["kwargs"].get("pooled_projections")
    timestep = inputs["kwargs"].get("timestep")

    capture.record("hidden_states_input", hidden_states)
    capture.record("encoder_hidden_states", encoder_hidden_states)
    if pooled_projections is not None:
        capture.record("pooled_projections", pooled_projections)

    print("\nCaptured input shapes:")
    print(capture.summary())

    # Warmup
    print(f"\nWarming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        with torch.no_grad():
            transformer(*inputs["args"], **inputs["kwargs"])
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling ({profile_iters} iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for i in range(profile_iters):
        start_event.record()
        with torch.no_grad():
            nvtx.range_push(f"transformer_forward_{i}")
            output = transformer(*inputs["args"], **inputs["kwargs"])
            nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    avg_time = sum(times) / len(times)
    print(f"Average transformer forward time: {avg_time:.2f} ms")

    return avg_time, capture


def print_shape_analysis(height: int = 1024, width: int = 1024):
    """Print expected shapes and bandwidth estimates for a resolution.

    Args:
        height: Image height in pixels
        width: Image width in pixels
    """
    shapes = get_flux_shapes_for_resolution(height, width)
    hidden_shape = shapes["hidden_states"]
    encoder_shape = shapes["encoder_hidden_states"]
    temb_shape = shapes["temb"]

    print(f"\nExpected Shapes for {height}x{width}:")
    print("-" * 60)

    # Calculate sizes (bf16 = 2 bytes)
    dtype_bytes = 2
    hidden_bytes = hidden_shape[0] * hidden_shape[1] * hidden_shape[2] * dtype_bytes
    encoder_bytes = encoder_shape[0] * encoder_shape[1] * encoder_shape[2] * dtype_bytes
    temb_bytes = temb_shape[0] * temb_shape[1] * dtype_bytes

    print(f"hidden_states:         {str(hidden_shape):30s} = {hidden_bytes / 1e6:.2f} MB")
    print(f"encoder_hidden_states: {str(encoder_shape):30s} = {encoder_bytes / 1e6:.2f} MB")
    print(f"temb:                  {str(temb_shape):30s} = {temb_bytes / 1e3:.2f} KB")

    # Estimate element-wise bytes per block
    elementwise = estimate_elementwise_bytes(hidden_shape, encoder_shape, temb_shape, dtype_bytes)

    print("\nPer-Operation Element-wise Estimates (read + write):")
    for op_name, bytes_val in elementwise.items():
        print(f"  {op_name:25s}: {bytes_val / 1e6:.2f} MB")

    # Per-layer estimate
    double_block_bytes = estimate_layer_elementwise_bytes(
        hidden_shape, encoder_shape, "double", dtype_bytes
    )
    single_block_bytes = estimate_layer_elementwise_bytes(
        hidden_shape, encoder_shape, "single", dtype_bytes
    )

    print(f"\nPer-Layer Element-wise Estimates:")
    print(f"  Double block: ~{double_block_bytes / 1e6:.1f} MB")
    print(f"  Single block: ~{single_block_bytes / 1e6:.1f} MB")

    # Total for full model (19 double + 38 single)
    total_elementwise = 19 * double_block_bytes + 38 * single_block_bytes
    print(f"\nTotal element-wise memory traffic (estimate):")
    print(f"  19 double + 38 single = ~{total_elementwise / 1e9:.2f} GB")


def analyze_nsys_database(db_path: str, peak_gbs: float = GPU_PEAK_BANDWIDTH["B200"]):
    """Analyze nsys SQLite database and print bandwidth report.

    Args:
        db_path: Path to nsys .sqlite export
        peak_gbs: Peak GPU bandwidth
    """
    from nunchaku.profiling import parse_nsys_sqlite
    from nunchaku.profiling.nsys_parser import (
        aggregate_kernel_timings,
        filter_elementwise_kernels,
        summarize_by_op_type,
    )

    print(f"\nAnalyzing nsys database: {db_path}")
    print("=" * 80)

    timings = parse_nsys_sqlite(db_path)
    print(f"Total kernels: {len(timings)}")

    # Summarize by operation type
    summary = summarize_by_op_type(timings)
    print("\nKernel Time by Operation Type:")
    for op_type, stats in sorted(summary.items(), key=lambda x: -x[1]["total_ns"]):
        print(f"  {op_type:15s}: {stats['total_ms']:8.2f} ms ({stats['pct']:5.1f}%)")

    # Focus on element-wise kernels
    elementwise = filter_elementwise_kernels(timings)
    print(f"\nElement-wise kernels: {len(elementwise)}")

    # Aggregate by kernel name
    aggregated = aggregate_kernel_timings(elementwise)
    print("\nElement-wise Kernel Breakdown:")
    for name, stats in sorted(aggregated.items(), key=lambda x: -x[1]["total_ns"]):
        total_ms = stats["total_ns"] * 1e-6
        avg_us = stats["avg_ns"] * 1e-3
        print(
            f"  {name:40s}: {total_ms:8.3f} ms total, "
            f"{avg_us:6.1f} us avg, {int(stats['count']):4d} calls"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Profile memory bandwidth for Nunchaku element-wise kernels"
    )
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
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=5,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--peak-bandwidth",
        type=float,
        default=GPU_PEAK_BANDWIDTH["B200"],
        help="Peak GPU memory bandwidth in GB/s",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=list(GPU_PEAK_BANDWIDTH.keys()),
        help="Select GPU type for peak bandwidth (overrides --peak-bandwidth)",
    )
    parser.add_argument(
        "--b200-mode",
        action="store_true",
        help="Enable B200 mode (stubs FP4 GEMM for element-wise profiling)",
    )
    parser.add_argument(
        "--layer-profile",
        action="store_true",
        help="Profile individual transformer layers using forward_layer_at",
    )
    parser.add_argument(
        "--sweep-shapes",
        action="store_true",
        help="Print shape analysis for different resolutions without running model",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="512,768,1024,1536,2048",
        help="Comma-separated list of resolutions for shape sweep",
    )
    parser.add_argument(
        "--nsys-db",
        type=str,
        help="Path to nsys SQLite database for kernel timing analysis",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed bandwidth report",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for report (default: stdout)",
    )
    # Kernel bandwidth mode arguments
    parser.add_argument(
        "--kernel-bandwidth",
        action="store_true",
        help="Calculate bandwidth for element-wise kernels using provided timings",
    )
    parser.add_argument(
        "--quantize-time",
        type=float,
        default=0.25969,
        help="Measured time for quantize kernels in seconds (default: 0.25969)",
    )
    parser.add_argument(
        "--layernorm-time",
        type=float,
        default=0.108147,
        help="Measured time for layernorm kernels in seconds (default: 0.108147)",
    )
    parser.add_argument(
        "--mul-add-time",
        type=float,
        default=0.105632,
        help="Measured time for mul_add kernels in seconds (default: 0.105632)",
    )
    parser.add_argument(
        "--add-time",
        type=float,
        default=0.099116,
        help="Measured time for add kernels in seconds (default: 0.099116)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50)",
    )
    parser.add_argument(
        "--img-tokens",
        type=int,
        default=None,
        help="Number of image tokens (default: calculated from height/width)",
    )
    parser.add_argument(
        "--txt-tokens",
        type=int,
        default=512,
        help="Number of text tokens (default: 512)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=3072,
        help="Hidden dimension size (default: 3072)",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=12288,
        help="FFN hidden dimension (default: 12288, typically 4x hidden_dim)",
    )
    # Actual kernel call counts from nsys (overrides estimates)
    parser.add_argument(
        "--quantize-calls",
        type=int,
        default=None,
        help="Actual quantize kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--layernorm-calls",
        type=int,
        default=None,
        help="Actual layernorm kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--mul-add-calls",
        type=int,
        default=None,
        help="Actual mul_add kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--add-calls",
        type=int,
        default=None,
        help="Actual add kernel calls from nsys (overrides estimate)",
    )
    parser.add_argument(
        "--capture-shapes",
        action="store_true",
        help="Run model to capture actual tensor shapes, then calculate bandwidth",
    )

    args = parser.parse_args()

    # Override peak bandwidth if GPU specified
    if args.gpu:
        args.peak_bandwidth = GPU_PEAK_BANDWIDTH[args.gpu]

    print("=" * 80)
    print("Memory Bandwidth Profiler for Nunchaku Element-wise Kernels")
    print(f"Peak Bandwidth: {args.peak_bandwidth:.0f} GB/s")
    print("=" * 80)

    # Shape sweep mode - no model loading needed
    if args.sweep_shapes:
        resolutions = [int(r) for r in args.resolutions.split(",")]
        for res in resolutions:
            print_shape_analysis(res, res)
        return

    # Nsys analysis mode
    if args.nsys_db:
        analyze_nsys_database(args.nsys_db, args.peak_bandwidth)
        return

    # Kernel bandwidth mode - calculate bandwidth from provided timings
    if args.kernel_bandwidth:
        # Calculate image tokens from resolution if not specified
        if args.img_tokens is not None:
            img_tokens = args.img_tokens
        else:
            # Flux uses 16x16 patches, so tokens = (height/16) * (width/16)
            img_tokens = (args.height // 16) * (args.width // 16)

        results, calls = calculate_kernel_bandwidths(
            img_tokens=img_tokens,
            txt_tokens=args.txt_tokens,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            batch=1,
            num_double_blocks=19,
            num_single_blocks=38,
            num_steps=args.num_steps,
            peak_gbs=args.peak_bandwidth,
            quantize_time_s=args.quantize_time,
            layernorm_time_s=args.layernorm_time,
            mul_add_time_s=args.mul_add_time,
            add_time_s=args.add_time,
            quantize_calls=args.quantize_calls,
            layernorm_calls=args.layernorm_calls,
            mul_add_calls=args.mul_add_calls,
            add_calls=args.add_calls,
        )

        # Check if any nsys counts were provided
        calls_from_nsys = any([
            args.quantize_calls is not None,
            args.layernorm_calls is not None,
            args.mul_add_calls is not None,
            args.add_calls is not None,
        ])

        report = generate_kernel_bandwidth_report(
            results,
            calls,
            peak_gbs=args.peak_bandwidth,
            img_tokens=img_tokens,
            txt_tokens=args.txt_tokens,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            num_steps=args.num_steps,
            calls_from_nsys=calls_from_nsys,
        )

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report written to: {args.output}")
        else:
            print(report)
        return

    # Architecture-based bandwidth calculation (no model loading needed)
    if args.capture_shapes:
        # Calculate image tokens for this resolution
        img_tokens = (args.height // 16) * (args.width // 16)

        print(f"\nCalculating bytes from Flux architecture...")
        print(f"  Image tokens: {img_tokens}, Text tokens: {args.txt_tokens}")
        print(f"  Hidden dim: {args.hidden_dim}, MLP hidden dim: {args.mlp_hidden_dim}")

        # Get architecture-based byte counts
        config = FluxModelConfig(
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            num_double_blocks=19,
            num_single_blocks=38,
            img_tokens=img_tokens,
            txt_tokens=args.txt_tokens,
        )
        kernel_data = calculate_flux_kernel_bytes(config, num_steps=1)

        print("\nKernel data (per forward pass):")
        for kernel, data in kernel_data.items():
            print(f"  {kernel}: {data['calls_per_step']} calls, {data['bytes_per_step'] / 1e6:.2f} MB")

        # Calculate bandwidth using architecture-based bytes + nsys timings
        results, calls = calculate_bandwidth_from_architecture(
            img_tokens=img_tokens,
            txt_tokens=args.txt_tokens,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            num_double_blocks=19,
            num_single_blocks=38,
            num_steps=args.num_steps,
            peak_gbs=args.peak_bandwidth,
            quantize_time_s=args.quantize_time,
            layernorm_time_s=args.layernorm_time,
            mul_add_time_s=args.mul_add_time,
            add_time_s=args.add_time,
        )

        report = generate_kernel_bandwidth_report(
            results,
            calls,
            peak_gbs=args.peak_bandwidth,
            img_tokens=img_tokens,
            txt_tokens=args.txt_tokens,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            num_steps=args.num_steps,
            calls_from_nsys=False,  # Bytes from architecture analysis
        )

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport written to: {args.output}")
        else:
            print("\n" + report)
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    print(f"Precision: {args.precision}")

    if args.b200_mode and args.precision == "fp4":
        cutils.set_stub_fp4_gemm(True)
        print("B200 mode: FP4 GEMM stub enabled")

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        args.model, precision=args.precision
    )

    from diffusers import FluxPipeline

    pipeline = FluxPipeline.from_pretrained(
        args.base_model, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    # Capture transformer inputs
    print(f"\nCapturing transformer inputs for {args.height}x{args.width}...")
    inputs = capture_transformer_inputs(
        pipeline, "test", height=args.height, width=args.width
    )

    # Print shape analysis
    print_shape_analysis(args.height, args.width)

    if args.layer_profile:
        # Profile individual layers
        hidden_states = inputs["kwargs"].get("hidden_states") or inputs["args"][0]
        encoder_hidden_states = inputs["kwargs"]["encoder_hidden_states"]

        # Get temb from the model
        timestep = inputs["kwargs"]["timestep"]
        pooled_projections = inputs["kwargs"]["pooled_projections"]
        guidance = inputs["kwargs"].get("guidance")

        # Compute embeddings as model does
        with torch.no_grad():
            hidden_states = transformer.x_embedder(hidden_states)
            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
                temb = transformer.time_text_embed(timestep, guidance, pooled_projections)
            else:
                temb = transformer.time_text_embed(timestep, pooled_projections)
            encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

            # Get rotary embeddings
            txt_ids = inputs["kwargs"]["txt_ids"]
            img_ids = inputs["kwargs"]["img_ids"]
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = transformer.pos_embed(ids)

        layer_results = profile_transformer_layers(
            transformer,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            num_double_blocks=19,
            warmup_iters=args.warmup_iters,
            peak_gbs=args.peak_bandwidth,
        )

        if args.report:
            report = generate_bandwidth_report(layer_results, peak_gbs=args.peak_bandwidth)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"\nReport written to: {args.output}")
            else:
                print("\n" + report)

    else:
        # Profile full transformer
        avg_time, capture = profile_full_transformer(
            transformer,
            inputs,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
            peak_gbs=args.peak_bandwidth,
        )

        # Estimate bandwidth for full forward
        hidden_states = inputs["kwargs"].get("hidden_states") or inputs["args"][0]
        encoder_hidden_states = inputs["kwargs"]["encoder_hidden_states"]
        hidden_shape = tuple(hidden_states.shape)
        encoder_shape = tuple(encoder_hidden_states.shape)
        dtype_bytes = 2  # bf16

        # Total element-wise bytes estimate
        double_block_bytes = estimate_layer_elementwise_bytes(
            hidden_shape, encoder_shape, "double", dtype_bytes
        )
        single_block_bytes = estimate_layer_elementwise_bytes(
            hidden_shape, encoder_shape, "single", dtype_bytes
        )
        total_bytes = 19 * double_block_bytes + 38 * single_block_bytes

        bw_result = calculate_bandwidth_from_ms(total_bytes, avg_time, args.peak_bandwidth)

        print(f"\nFull Transformer Bandwidth Estimate:")
        print(f"  Total element-wise bytes: ~{total_bytes / 1e9:.2f} GB")
        print(f"  Time: {avg_time:.2f} ms")
        print(f"  Effective bandwidth: ~{bw_result.bandwidth_gbs:.0f} GB/s")
        print(f"  Efficiency: {bw_result.efficiency_pct:.1f}%")
        print()
        print("Note: This is an upper bound estimate. Actual element-wise bandwidth")
        print("is lower since GEMM kernels dominate execution time.")

    print("\nProfiling complete!")
    print("\nTo get detailed kernel timings, run with nsys:")
    print(f"  nsys profile -o nunchaku_bw --export sqlite python {__file__}")
    print(f"  python {__file__} --nsys-db nunchaku_bw.sqlite")


if __name__ == "__main__":
    main()
