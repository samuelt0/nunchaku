"""
Profile effective memory bandwidth (bytes moved / time) for elementwise-ish ops
in Nunchaku/Flux on B200 with FP4 GEMM stubbed.

This script:
  1) Runs a real inference (transformer-only or full pipeline)
  2) Uses torch.profiler to collect CUDA time per op, grouped by input shapes
  3) Uses TorchDispatchMode to estimate bytes read/written per op call
  4) Joins them to compute effective bandwidth per op+shape signature

Note:
- This is "algorithmic / effective" bandwidth, not true DRAM bytes.
- Later, use NCU metrics for per-kernel DRAM throughput.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import ProfilerActivity

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku._C import utils as cutils


def _normalize_op_key(func) -> str:
    """
    Convert a torch dispatch func (e.g. aten.add.Tensor) into something close
    to torch.profiler keys (e.g. aten::add).
    """
    s = str(func)  # e.g. "aten.add.Tensor" or "nunchaku::something"
    # Handle OpOverload strings like "aten.add.Tensor"
    if s.startswith("aten."):
        # "aten.add.Tensor" -> "aten::add"
        core = s.split(".", 2)[1]  # add
        return f"aten::{core}"
    # Some custom ops may look like "nunchaku::quantize_fp4" already
    return s


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _collect_tensors(obj: Any, out: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _collect_tensors(x, out)
    return out


def _input_shape_sig(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[int, ...], ...]:
    """
    Shape signature for grouping:
    tuple of shapes of tensor inputs (kwargs included), in encounter order.
    """
    shapes: List[Tuple[int, ...]] = []
    for t in _collect_tensors(args):
        shapes.append(tuple(t.shape))
    for t in _collect_tensors(kwargs):
        shapes.append(tuple(t.shape))
    return tuple(shapes)


@dataclass
class BytesAgg:
    calls: int = 0
    in_bytes: int = 0
    out_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.in_bytes + self.out_bytes


class OpBytesRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    """
    Records bytes read/written for selected ops, grouped by (op_key, input_shape_sig).

    We count:
      - inputs: unique tensor storages (by data_ptr) to avoid double-counting same tensor arg
      - outputs: all output tensors returned (including aux outputs like mean/rstd)

    This is an *estimate* of algorithmic bytes, not actual DRAM bytes.
    """

    def __init__(self, targets_exact: List[str], targets_substr: List[str]):
        super().__init__()
        self.targets_exact = set(targets_exact)
        self.targets_substr = list(targets_substr)
        self.bytes_by_key: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], BytesAgg] = {}

    def _is_target(self, op_key: str) -> bool:
        if op_key in self.targets_exact:
            return True
        for sub in self.targets_substr:
            if sub in op_key:
                return True
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        op_key = _normalize_op_key(func)
        track = self._is_target(op_key)
        sig = _input_shape_sig(args, kwargs) if track else None

        # Execute the op first (so we can see outputs)
        out = func(*args, **kwargs)

        if track and sig is not None:
            # Inputs: unique tensor buffers
            in_tensors = _collect_tensors(args) + _collect_tensors(kwargs)
            seen_ptrs = set()
            in_bytes = 0
            for t in in_tensors:
                # Some tensors may be meta/undefined, be defensive
                try:
                    ptr = int(t.data_ptr())
                except Exception:
                    ptr = id(t)
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                in_bytes += _tensor_nbytes(t)

            # Outputs: count all tensor outputs
            out_tensors = _collect_tensors(out)
            out_bytes = sum(_tensor_nbytes(t) for t in out_tensors)

            k = (op_key, sig)
            agg = self.bytes_by_key.get(k)
            if agg is None:
                agg = BytesAgg()
                self.bytes_by_key[k] = agg
            agg.calls += 1
            agg.in_bytes += in_bytes
            agg.out_bytes += out_bytes

        return out


def _get_cuda_time_total_us(evt) -> float:
    """
    Torch profiler API differs across versions.
    Try common attribute names for total GPU/CUDA time.
    Returns microseconds.
    """
    # Most common newer API:
    if hasattr(evt, "cuda_time_total"):
        return float(evt.cuda_time_total)
    # Some builds use "device_time_total" for CUDA time:
    if hasattr(evt, "device_time_total"):
        return float(evt.device_time_total)
    # Some expose "self_cuda_time_total" (self time only, but better than nothing):
    if hasattr(evt, "self_cuda_time_total"):
        return float(evt.self_cuda_time_total)
    # Some use "self_device_time_total":
    if hasattr(evt, "self_device_time_total"):
        return float(evt.self_device_time_total)
    return 0.0


def _extract_prof_times_us(prof) -> Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Tuple[int, float]]:
    """
    Returns mapping:
      (op_key, input_shape_sig) -> (calls, cuda_time_total_us)
    """
    out: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Tuple[int, float]] = {}
    for evt in prof.key_averages(group_by_input_shape=True):
        key = evt.key
        shapes = tuple(tuple(s) for s in (evt.input_shapes or []))
        calls = int(getattr(evt, "count", 0))
        cuda_us = _get_cuda_time_total_us(evt)
        if cuda_us <= 0:
            continue
        out[(key, shapes)] = (calls, cuda_us)
    return out



def _print_report(
    bytes_rec: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], BytesAgg],
    times_us: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Tuple[int, float]],
    peak_gbps: Optional[float],
    top_k: int,
):
    rows = []
    keys = set(bytes_rec.keys()) | set(times_us.keys())
    for k in keys:
        op, sig = k
        b = bytes_rec.get(k, BytesAgg())
        calls_t, cuda_us = times_us.get(k, (0, 0.0))
        calls = max(b.calls, calls_t)
        if cuda_us <= 0:
            continue
        t_s = cuda_us * 1e-6
        bw_gbs = (b.total_bytes / 1e9) / t_s if b.total_bytes > 0 else 0.0
        eff = (bw_gbs / peak_gbps) if (peak_gbps and peak_gbps > 0) else None
        rows.append(
            (cuda_us, bw_gbs, eff, calls, b.in_bytes, b.out_bytes, op, sig)
        )

    rows.sort(key=lambda x: x[0], reverse=True)  # sort by total CUDA time (desc)

    print("\n=== Effective Bandwidth Report (algorithmic bytes / CUDA time) ===")
    if peak_gbps:
        print(f"Peak HBM BW (provided): {peak_gbps:.1f} GB/s")
    print(f"{'CUDA ms':>9}  {'BW (GB/s)':>10}  {'Eff':>7}  {'Calls':>7}  {'Read(MB)':>9}  {'Write(MB)':>10}  Op")
    print("-" * 120)

    for i, (cuda_us, bw_gbs, eff, calls, in_b, out_b, op, sig) in enumerate(rows[:top_k]):
        cuda_ms = cuda_us / 1000.0
        read_mb = in_b / 1e6
        write_mb = out_b / 1e6
        eff_str = f"{eff*100:5.1f}%" if eff is not None else "  n/a "
        # keep sig short
        sig_str = ""
        if sig:
            # show up to first 2 tensor shapes
            shown = list(sig[:2])
            sig_str = f" shapes={shown}" + (" ..." if len(sig) > 2 else "")
        print(f"{cuda_ms:9.3f}  {bw_gbs:10.1f}  {eff_str:>7}  {calls:7d}  {read_mb:9.1f}  {write_mb:10.1f}  {op}{sig_str}")

    print("\nNotes:")
    print(" - Bytes are estimated from tensor sizes at the op boundary (algorithmic bytes).")
    print(" - CUDA time comes from torch.profiler, which may include multiple kernels per op.")
    print(" - Later: use NCU to get true DRAM/L2 bytes per individual kernel.\n")


def main():
    parser = argparse.ArgumentParser(description="Profile effective memory bandwidth for elementwise ops (B200)")
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
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--transformer-only", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--peak-hbm-gbps",
        type=float,
        default=0.0,
        help="If provided, prints efficiency = BW_eff / peak (GB/s).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Precision: {args.precision}")

    if args.precision == "fp4":
        cutils.set_stub_fp4_gemm(True)
        print("FP4 GEMM stub mode enabled")

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        args.model, precision=args.precision
    )

    from diffusers import FluxPipeline

    pipeline = FluxPipeline.from_pretrained(
        args.base_model, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    # ---- Target ops (you can expand this list as you discover names) ----
    # Exact matches (torch.profiler style keys)
    targets_exact = [
        # Add / mul / fused mul-add-ish
        "aten::add",
        "aten::add_",
        "aten::mul",
        "aten::mul_",
        "aten::addcmul",
        "aten::addcmul_",
        # Layernorm variants (different backends show different keys)
        "aten::layer_norm",
        "aten::native_layer_norm",
    ]

    # Substring matches for quantization ops (covers custom ops + aten quantize variants)
    targets_substr = [
        "quant", "fp4", "int4", "svdq", "nunchaku::", "pack", "dequant"
    ]

    # --- Optional: capture transformer inputs once, like your original script ---
    inputs = {}

    if args.transformer_only:
        print("Capturing transformer inputs...")

        def capture_hook(module, hook_args, hook_kwargs):
            inputs["args"] = hook_args
            inputs["kwargs"] = hook_kwargs

        pipeline.transformer.register_forward_pre_hook(capture_hook, with_kwargs=True)

        with torch.no_grad():
            pipeline(
                "test",
                num_inference_steps=1,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                output_type="latent",
            )

    # Warmup (no profiling)
    print(f"Warming up ({args.warmup_steps} iterations)...")
    for _ in range(args.warmup_steps):
        with torch.no_grad():
            if args.transformer_only:
                pipeline.transformer(*inputs["args"], **inputs["kwargs"])
            else:
                pipeline(
                    "test",
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    output_type="latent",
                )
    torch.cuda.synchronize()

    # Profiling run: torch.profiler for CUDA time + TorchDispatchMode for bytes
    recorder = OpBytesRecorder(targets_exact=targets_exact, targets_substr=targets_substr)

    print(f"Profiling ({args.profile_steps} iterations)...")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        with recorder:
            nvtx.range_push("bw_profile")
            for i in range(args.profile_steps):
                nvtx.range_push(f"iter_{i}")
                with torch.no_grad():
                    if args.transformer_only:
                        pipeline.transformer(*inputs["args"], **inputs["kwargs"])
                    else:
                        pipeline(
                            "A cat sitting on a windowsill",
                            num_inference_steps=args.num_inference_steps,
                            height=args.height,
                            width=args.width,
                            guidance_scale=args.guidance_scale,
                            output_type="latent",
                        )
                nvtx.range_pop()
            nvtx.range_pop()

    torch.cuda.synchronize()

    times_us = _extract_prof_times_us(prof)

    peak = args.peak_hbm_gbps if args.peak_hbm_gbps > 0 else None
    _print_report(recorder.bytes_by_key, times_us, peak, top_k=args.top_k)

    print("Done.")
    print("Tip: If you don’t see your quantize op yet, print unique op names from the profiler")
    print("     and add them to targets_exact / targets_substr.")


if __name__ == "__main__":
    main()
