#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""FP8 PTQ for System 1: quantize traj_dit and the memory block, export ONNX, build engines.

System 1 ships BF16 by default. This is the experiment that asks whether it should not.

The route differs from System 2's. System 2 goes through TensorRT-Edge-LLM, which consumes a
quantized HF checkpoint and inserts the scaling itself. System 1 is a plain
``torch.onnx.export`` into ``trtexec``, and **TensorRT supports FP8 only through explicit
quantization** -- there is no implicit FP8 calibration the way there is for INT8. So the Q/DQ
nodes have to be in the ONNX, which means a ModelOpt PTQ pass in PyTorch first.

Calibration comes from ``dump_system1_calib.py`` rather than from random tensors, and that
matters more here than usual: FP8 scale factors are amax-based, and both of these modules
consume tensors whose scale is set by something upstream -- ``z_latents`` by System 2 through
``cond_projector``, the frames by the ResNet normalization. A synthetic draw has the wrong
amax and produces wrong scales, quietly.

The calibration bundle is read from disk rather than regenerated because capturing it runs
System 2 on every sample; caching it means re-quantizing with a different config costs a
minute instead of the whole pipeline.

Run under the transformers 4.51 environment (Python 3.10 here), which has both ModelOpt and
a working InternNav::

    INTERNNAV_PATH=~/InternNav PYTHONPATH=~/InternNav \\
    python quantize_system1.py --calib_path work/system1_calib.pt --out_dir work/onnx_fp8
"""
import argparse
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import internvla_compat  # noqa: E402
from memblock import MemBlock  # noqa: E402

TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--internvla_ckpt",
                   default=os.path.expanduser(
                       os.environ.get("INTERNVLA_CKPT",
                                      "~/InternNav/checkpoints/InternVLA-N1-DualVLN")))
    p.add_argument("--calib_path", required=True,
                   help="Bundle from dump_system1_calib.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--components", default="traj_dit,memory",
                   help="Comma-separated subset of traj_dit,memory")
    p.add_argument("--zlen", type=int, default=36,
                   help="Optimization profile z_latents length (32 memory tokens + 4 TRAJ)")
    p.add_argument("--exclude_memory", default="nn.Conv2d",
                   help="Comma-separated exclusions for the memory block: 'nn.X' matches a "
                        "module class, anything else is a name glob. The default leaves "
                        "Conv2d alone because the legacy ONNX exporter cannot infer a "
                        "convolution kernel shape through Q/DQ and dies with 'convolution "
                        "for kernel of unknown shape'. In DepthAnythingV2 that is the "
                        "patch_embed projection -- one layer of a ViT, so almost no compute "
                        "is left behind.")
    p.add_argument("--skip_build", action="store_true",
                   help="Export ONNX only, do not call trtexec")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def fp8_config(exclude: str = ""):
    """FP8_DEFAULT_CFG with name patterns disabled."""
    import copy

    import modelopt.torch.quantization as mtq

    # In ModelOpt 0.44 quant_cfg is an ordered *list* of rules, not a dict, and later rules
    # win -- so exclusions go on the end. Entries starting with "nn." match by module class
    # (the form the built-in BatchNorm exclusions use); anything else matches by name glob.
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pat in (e.strip() for e in exclude.split(",")):
        if not pat:
            continue
        rule = {"quantizer_name": "*", "enable": False}
        if pat.startswith("nn."):
            rule["parent_class"] = pat
        else:
            rule["quantizer_name"] = pat
        cfg["quant_cfg"].append(rule)
    return cfg


def count_qdq(onnx_path: str) -> tuple[int, int]:
    """Q/DQ node counts. Zero means the PTQ pass did not survive the export, and the
    engine below would silently be BF16 with extra steps.

    ModelOpt emits TensorRT's own ``trt::TRT_FP8QuantizeLinear`` for FP8 rather than the
    standard ONNX ``QuantizeLinear``, so counting only the latter reports a correctly
    quantized graph as unquantized."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    ops = [n.op_type for n in model.graph.node]
    q = ops.count("QuantizeLinear") + ops.count("TRT_FP8QuantizeLinear")
    dq = ops.count("DequantizeLinear") + ops.count("TRT_FP8DequantizeLinear")
    return q, dq


def build(onnx_path: str, engine_path: str, shapes: list[str]) -> bool:
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp8", "--bf16"]
    cmd += shapes
    print("      " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(engine_path):
        tail = "\n      ".join((r.stdout + r.stderr).strip().splitlines()[-12:])
        print(f"      BUILD FAILED\n      {tail}")
        return False
    print(f"      engine {os.path.getsize(engine_path) / 1e6:.1f} MB")
    return True


def quantize_traj_dit(inner, calib, args) -> str:
    import modelopt.torch.quantization as mtq

    dit = inner.traj_dit.eval()
    batches = calib["dit_batches"]
    dev = args.device

    def forward_loop(m):
        for x, ts, z in batches:
            with torch.no_grad():
                m(x=x.to(dev, torch.bfloat16), timestep=ts.to(dev),
                  z_latents=z.to(dev, torch.bfloat16))

    print(f"[traj_dit] PTQ FP8 over {len(batches)} real batches")
    dit = mtq.quantize(dit, fp8_config(), forward_loop=forward_loop)

    # Export in FP32: the RoPE freqs_cis are float32 and a mixed-precision trace fails.
    # Precision is re-established by the Q/DQ nodes plus trtexec --bf16.
    dit = dit.to(torch.float32).eval()
    x, ts, z = batches[0]
    x, ts, z = x.to(dev).float(), ts.to(dev), z.to(dev).float()

    class Wrap(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, x, timestep, z_latents):
            return self.d(x=x, timestep=timestep, z_latents=z_latents)

    w = Wrap(dit).eval()
    onnx_path = os.path.join(args.out_dir, "system1_traj_dit_fp8.onnx")
    dyn = {"x": {0: "batch"}, "timestep": {0: "batch"},
           "z_latents": {0: "batch", 1: "zlen"}, "output": {0: "batch"}}
    print(f"[traj_dit] Export ONNX -> {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(w, (x, ts, z), onnx_path,
                          input_names=["x", "timestep", "z_latents"],
                          output_names=["output"], opset_version=19,
                          do_constant_folding=True, export_params=True,
                          dynamic_axes=dyn, dynamo=False)
    q, dq = count_qdq(onnx_path)
    print(f"      {os.path.getsize(onnx_path) / 1e6:.1f} MB, "
          f"{q} QuantizeLinear / {dq} DequantizeLinear")
    if q == 0:
        print("      [WARN] no Q/DQ in the graph -- the engine will not be FP8")

    if args.skip_build:
        return onnx_path
    B, WP, DIM = x.shape[0], x.shape[1], x.shape[2]
    zdim = z.shape[-1]
    engine = os.path.join(args.out_dir, "system1_traj_dit_fp8.engine")
    print("[traj_dit] Build engine")
    build(onnx_path, engine, [
        f"--minShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x4x{zdim}",
        f"--optShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x{args.zlen}x{zdim}",
        f"--maxShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x64x{zdim}"])
    return onnx_path


def quantize_memory(model, inner, calib, args) -> str:
    import modelopt.torch.quantization as mtq

    dev = args.device
    block = MemBlock(inner.rgb_model, inner.memory_encoder, inner.rgb_resampler).eval()
    batches = calib["image_batches"]

    def forward_loop(m):
        for imgs in batches:
            with torch.no_grad():
                m(imgs.to(dev, torch.bfloat16))

    print(f"[memory] PTQ FP8 over {len(batches)} real frame batches"
          + (f", excluding {args.exclude_memory}" if args.exclude_memory else ""))
    block = mtq.quantize(block, fp8_config(args.exclude_memory), forward_loop=forward_loop)
    block = block.to(torch.float32).eval()

    imgs = batches[0].to(dev).float()
    onnx_path = os.path.join(args.out_dir, "system1_memory_fp8.onnx")
    print(f"[memory] Export ONNX -> {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(block, (imgs,), onnx_path, input_names=["images"],
                          output_names=["memory_tokens"], opset_version=19,
                          do_constant_folding=True, export_params=True,
                          dynamic_axes={"images": {0: "frames"}}, dynamo=False)
    q, dq = count_qdq(onnx_path)
    print(f"      {os.path.getsize(onnx_path) / 1e6:.1f} MB, "
          f"{q} QuantizeLinear / {dq} DequantizeLinear")
    if q == 0:
        print("      [WARN] no Q/DQ in the graph -- the engine will not be FP8")

    if args.skip_build:
        return onnx_path
    T, C, H, W = imgs.shape
    engine = os.path.join(args.out_dir, "system1_memory_fp8.engine")
    print("[memory] Build engine")
    build(onnx_path, engine, [
        f"--minShapes=images:1x{C}x{H}x{W}",
        f"--optShapes=images:{T}x{C}x{H}x{W}",
        f"--maxShapes=images:8x{C}x{H}x{W}"])
    return onnx_path


def main() -> int:
    args = parse_args()
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    # The memory block's QFormer takes PyTorch's fused MHA fast path, and
    # aten::_transformer_encoder_layer_fwd has no ONNX symbolic. Same toggle the BF16
    # export uses.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception as exc:                                  # pragma: no cover
        print(f"  (mha fastpath toggle unavailable: {exc})")
    os.makedirs(args.out_dir, exist_ok=True)
    internvla_compat.apply_all(need_system1=True, allow_missing_depth=False)

    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)

    calib = torch.load(args.calib_path, map_location="cpu", weights_only=False)
    print(f"Loading {args.internvla_ckpt}")
    config = InternVLAN1ModelConfig.from_pretrained(args.internvla_ckpt)
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.internvla_ckpt, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device).eval()
    inner = model.get_model()

    if "traj_dit" in wanted:
        quantize_traj_dit(inner, calib, args)
    if "memory" in wanted:
        quantize_memory(model, inner, calib, args)
    print("\nDone. Verify with verify/compare_system1_engines.py --engine_dir "
          f"{args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
