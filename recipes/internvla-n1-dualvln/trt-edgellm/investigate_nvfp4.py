#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Find out why NVFP4 keeps the text fluent but collapses the System 1 bridge.

NVFP4 quantizes, exports and generates coherent text, yet z_latents cosine against the
FP32 reference falls to 0.647 where FP8 holds 0.9956. That is not a paradox and this
script does not treat it as one -- the two readouts have very different sensitivity:

* Tokens are ``argmax(lm_head(final_norm(h)))`` over a 152k vocabulary. RMSNorm removes
  per-token scale entirely and argmax only cares about ranking, so a hidden state can be
  badly distorted and still select the same token.
* z_latents are ``cond_projector(final_norm(h[-4:]))`` -- a two-layer MLP with a GELU,
  consumed as a continuous 4x768 vector by a diffusion model. GELU is not scale-invariant
  and cosine over 3072 numbers has no ranking slack.

So the expectation is that the hidden states are genuinely damaged and argmax is simply
tolerant. The job here is to locate *where* and decide whether it is recoverable, not to
explain the contradiction away.

Stages, each appending to one JSON report:

  control   Is this even a quantization problem? Compare ModelOpt fake-quant NVFP4 in
            PyTorch against the NVFP4 engine, both against FP32. If fake-quant is fine and
            only the engine is broken, this is a TensorRT miscompile like the fc_h_fusion
            case, and the rest of these stages are the wrong investigation. Run this first:
            it is cheap and it decides which of two very different hunts to run.
  layers    Per-layer hidden-state cosine, FP32 vs FP8 vs NVFP4. Smooth decay means a
            global capacity limit; a knee at particular layers names the culprits.
  channels  At the last layer, how much of the error sits in the few high-magnitude
            channels. If masking <=1% of channels restores cosine > 0.99, this is
            outlier clipping under per-16-block scaling and AWQ or a targeted exclusion
            should fix it. If the error is uniform, no scaling trick will help.
  weightonly  NVFP4 is W4A4. Quantize weights only and leave activations in bf16. This is
            the single highest-information measurement: 4-bit *activations* through a
            3584-wide hidden state in blocks of 16 is the most likely culprit, and if
            W4A16 restores the bridge the remedy is immediate.

Decision rule, stated up front so the investigation is bounded: if any variant reaches
z_latents >= 0.99 it gets promoted to a supported scheme. If the best is 0.95-0.99 it stays
experimental with the number recorded. If nothing clears 0.95 after these stages, record
that NVFP4 is rejected on this model and stop -- do not keep going.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "quantize"))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine in float64. float32 accumulation over millions of elements is not enough --
    it silently returns values above 1.0 on tensors this size."""
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def load_bridge(ckpt: str, device: str):
    from safetensors import safe_open

    path = os.path.join(ckpt, "bridge.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found; run repackage_system2.py, which sets the bridge tensors "
            f"aside so this does not need the 16 GB original checkpoint")
    with safe_open(path, framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    cond = {k.replace("model.cond_projector.", ""): v.float().to(device)
            for k, v in tensors.items() if "cond_projector" in k}
    latent = tensors["model.latent_queries"].to(device)
    return latent, cond


def project(hidden: torch.Tensor, cond: dict) -> torch.Tensor:
    """The host-side bridge: linear, GELU, linear. Mirrors the deployed agent."""
    x = torch.nn.functional.linear(hidden.float(), cond["0.weight"], cond.get("0.bias"))
    x = torch.nn.functional.gelu(x)
    return torch.nn.functional.linear(x, cond["2.weight"], cond.get("2.bias"))


def fake_quantize(model, scheme: str, strategy: str, calib_texts, tokenizer, device: str):
    """Apply ModelOpt fake quantization in place and return the model."""
    import modelopt.torch.quantization as mtq
    from quant_schemes import build_quant_config

    quant_cfg = build_quant_config(scheme, strategy)
    enc = tokenizer(calib_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512)

    def forward_loop(m):
        for i in range(enc["input_ids"].shape[0]):
            m(enc["input_ids"][i:i + 1].to(device))

    return mtq.quantize(model, quant_cfg, forward_loop=forward_loop)


def stage_control(args, report: dict) -> None:
    """Is the collapse in the quantization, or only in the engine?"""
    print("\n=== stage: control — fake-quant PyTorch vs engine ===")
    print("If fake-quant is ~0.99 and only the engine is broken, this is a TensorRT")
    print("miscompile like fc_h_fusion, and the remaining stages are the wrong hunt.\n")
    report["control"] = {
        "status": "not_run",
        "note": "requires an NVFP4 checkpoint; run quantize.py --scheme nvfp4_default "
                "--strategy s1 --allow_experimental first",
    }


def stage_layers(args, report: dict) -> None:
    print("\n=== stage: layers — where does divergence start? ===")
    report["layers"] = {"status": "not_run"}


def stage_channels(args, report: dict) -> None:
    print("\n=== stage: channels — is the error concentrated in outliers? ===")
    report["channels"] = {"status": "not_run"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repkg_ckpt", required=True,
                   help="Repackaged System 2 checkpoint (the FP32/BF16 reference)")
    p.add_argument("--nvfp4_ckpt", default=None, help="NVFP4-quantized checkpoint")
    p.add_argument("--fp8_ckpt", default=None, help="FP8 checkpoint, for a middle datapoint")
    p.add_argument("--calib_data_root", default=None)
    p.add_argument("--work_dir", default=os.path.expanduser("~/vln-opt-work"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", default="all",
                   choices=["all", "control", "layers", "channels", "weightonly"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = os.path.join(args.work_dir, "nvfp4_investigation.json")
    report = {}
    if os.path.isfile(out):
        with open(out) as f:
            report = json.load(f)

    stages = {"control": stage_control, "layers": stage_layers, "channels": stage_channels}
    todo = list(stages) if args.stage == "all" else [args.stage]
    for name in todo:
        if name in stages:
            stages[name](args, report)

    os.makedirs(args.work_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
