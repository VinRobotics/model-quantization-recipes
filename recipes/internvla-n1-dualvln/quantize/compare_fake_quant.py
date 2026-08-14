#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Prove — or refute — that activation quantization is what the engine loses.

Three numbers exist for each scheme and until now only two were measured directly:

  weights only   weights quantized then reconstructed, activations bf16
  fake quant     weights AND activations quantized in PyTorch  <- this script
  engine         the TensorRT engine

The claim this recipe makes is that the gap between "weights only" and "engine" is
activation quantization rather than anything about the export or the runtime. That has so
far been an inference from three numbers, including the FP16 engine measuring 0.999471,
which bounds TensorRT's own cost at about 0.0005.

Measuring fake quant closes it directly. If fake quant lands near the engine, the claim
holds. If it lands near weights-only instead, the claim is wrong and the loss is somewhere
in the export or the runtime, which is a different investigation.

The comparison runs on the same z_latents bridge the rest of the recipe is gated on.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_builder as pb  # noqa: E402
from benchmark_accuracy import discover_samples  # noqa: E402
from load_quantized import load_fake_quant, load_for_eval  # noqa: E402

PROMPT = ("You are an autonomous navigation assistant. Your task is to go to the kitchen. "
          "Where should you go next to stay on track?")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def bridge_tensors(ckpt: str, device: str):
    from safetensors import safe_open

    path = os.path.join(ckpt, "bridge.safetensors")
    with safe_open(path, framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}
    return {k.replace("model.cond_projector.", ""): v.float().to(device)
            for k, v in raw.items() if "cond_projector" in k}


def z_latents(model, processor, cond, device: str) -> torch.Tensor:
    enc = processor.tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1][0, -4:].float()
    x = torch.nn.functional.linear(h, cond["0.weight"], cond.get("0.bias"))
    x = torch.nn.functional.gelu(x)
    return torch.nn.functional.linear(x, cond["2.weight"], cond.get("2.bias")).cpu()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repkg_ckpt", required=True)
    p.add_argument("--quant_ckpt", required=True, help="The PTQ checkpoint, for weights-only")
    p.add_argument("--scheme", default="nvfp4_default")
    p.add_argument("--strategy", default="s1")
    p.add_argument("--calib_data_root", required=True)
    p.add_argument("--num_calib_samples", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_path", default=None)
    args = p.parse_args()

    cond = bridge_tensors(args.repkg_ckpt, args.device)

    print("[1/3] reference (unquantized)")
    ref_model, ref_proc, _ = load_for_eval(args.repkg_ckpt, device=args.device)
    z_ref = z_latents(ref_model, ref_proc, cond, args.device)
    del ref_model
    torch.cuda.empty_cache()

    print("[2/3] weights only (activations left in bf16)")
    wo_model, wo_proc, _ = load_for_eval(args.quant_ckpt, device=args.device)
    z_wo = z_latents(wo_model, wo_proc, cond, args.device)
    del wo_model
    torch.cuda.empty_cache()

    print(f"[3/3] fake quant ({args.scheme}, weights AND activations)")
    calib = discover_samples(args.calib_data_root, args.num_calib_samples, seed=0)
    fq_model, _, fq_proc = load_fake_quant(
        args.repkg_ckpt, args.scheme, args.strategy, calib,
        prompt_builder=pb.build_sample_inputs, device=args.device)
    z_fq = z_latents(fq_model, fq_proc, cond, args.device)
    del fq_model
    torch.cuda.empty_cache()

    result = {
        "weights_only": cos(z_ref, z_wo),
        "fake_quant": cos(z_ref, z_fq),
    }
    print("\n=== z_latents vs the unquantized reference ===")
    print(f"  weights only : {result['weights_only']:.6f}")
    print(f"  fake quant   : {result['fake_quant']:.6f}   <- comparable to the engine")
    print("\n  Compare 'fake quant' against the engine number for this scheme. Close means")
    print("  activation quantization explains the engine's loss; far means it does not and")
    print("  the export or runtime is worth investigating instead.")

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
