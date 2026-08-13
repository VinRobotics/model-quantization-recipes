#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Find out why NVFP4 keeps the text fluent but collapses the System 1 bridge.

NVFP4 quantizes, exports and generates coherent text, yet z_latents cosine against the
reference falls to 0.647 where FP8 holds 0.99. That is not a paradox, and this script does
not treat it as one -- the two readouts have very different sensitivity:

* Tokens are ``argmax(lm_head(final_norm(h)))`` over a 152k vocabulary. RMSNorm removes
  per-token scale entirely and argmax only cares about ranking, so a hidden state can be
  badly distorted and still select the same token.
* z_latents are ``cond_projector(final_norm(h[-4:]))`` -- two linears with a GELU between,
  consumed as a continuous 4x768 vector by a diffusion model. GELU is not scale-invariant
  and cosine over 3072 numbers has no ranking slack.

So the expectation is that the hidden states are genuinely damaged and argmax is simply
tolerant. The job is to locate the damage and decide whether it is recoverable.

Stages, cheapest and most decisive first:

  weights   How much error does NVFP4 put into the weights, versus FP8? Pure checkpoint
            arithmetic, no forward pass. If NVFP4 weight error is only modestly worse than
            FP8's while the bridge is 50x worse, the collapse is not raw weight precision
            and the later stages matter. Runs in seconds.
  layers    Per-layer hidden-state cosine against the unquantized reference. Smooth decay
            means a global capacity limit and nothing local to exclude; a knee at specific
            layers names them.
  channels  At the last layer, how concentrated the error is. If masking the top ~1% of
            channels by magnitude restores cosine above 0.99, this is outlier clipping
            under per-16-element block scaling, and AWQ or a targeted exclusion should fix
            it. If the error is spread evenly, no scaling trick will help.
  bridge    End-to-end z_latents for each checkpoint, the number that actually decides.

Decision rule, fixed in advance so the investigation is bounded: any variant reaching
z_latents >= 0.99 is promoted to a supported scheme. A best of 0.95-0.99 stays experimental
with the number recorded. If nothing clears 0.95, record NVFP4 as rejected on this model
and stop -- do not keep hunting.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "quantize"))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine in float64.

    float32 accumulation over tensors this size silently returns values above 1.0, which
    is how a broken measurement can look like a good one.
    """
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def rel_err(ref: torch.Tensor, other: torch.Tensor) -> float:
    ref = ref.double()
    return float((other.double() - ref).norm() / ref.norm())


def stage_weights(args, report: dict) -> None:
    """Compare per-projection weight error, NVFP4 vs FP8, against the unquantized weights."""
    from load_quantized import dequantize_state_dict
    from safetensors import safe_open
    import glob

    print("\n=== stage: weights — how much error does each format put in the weights? ===")

    keep = ("layers.0.", "layers.13.", "layers.27.")
    base = {}
    for shard in sorted(glob.glob(os.path.join(args.repkg_ckpt, "*.safetensors"))):
        if os.path.basename(shard) == "bridge.safetensors":
            continue
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith(".weight") and any(x in k for x in keep) and "proj" in k:
                    base[k] = f.get_tensor(k)

    rows = {}
    for label, path in (("fp8", args.fp8_ckpt), ("nvfp4", args.nvfp4_ckpt)):
        if not path:
            continue
        state = dequantize_state_dict(path)
        errs, coss = [], []
        for k, ref in base.items():
            if k in state:
                errs.append(rel_err(ref, state[k]))
                coss.append(cos(ref, state[k]))
        rows[label] = {"n": len(errs),
                       "rel_err_mean": float(np.mean(errs)),
                       "cos_mean": float(np.mean(coss))}
        print(f"  {label:6s} over {len(errs)} projections: "
              f"rel-err {100 * np.mean(errs):.2f}%  cos {np.mean(coss):.6f}")

    if "fp8" in rows and "nvfp4" in rows:
        ratio = rows["nvfp4"]["rel_err_mean"] / max(rows["fp8"]["rel_err_mean"], 1e-12)
        print(f"\n  NVFP4 weight error is {ratio:.1f}x FP8's.")
        print("  Compare that against the bridge gap in the 'bridge' stage: if the bridge")
        print("  degrades far more than this ratio, raw weight precision is not the cause.")
        rows["nvfp4_over_fp8"] = ratio
    report["weights"] = rows


def _load(path, device):
    from load_quantized import load_for_eval
    return load_for_eval(path, device=device)


def stage_layers(args, report: dict) -> None:
    """Per-layer hidden-state cosine against the unquantized model, on one real prompt."""
    print("\n=== stage: layers — where does the divergence start? ===")

    prompt = ("You are an autonomous navigation assistant. Your task is to go to the "
              "kitchen. Where should you go next to stay on track?")

    hiddens = {}
    for label, path in (("ref", args.repkg_ckpt), ("fp8", args.fp8_ckpt),
                        ("nvfp4", args.nvfp4_ckpt)):
        if not path:
            continue
        model, processor, _ = _load(path, args.device)
        enc = processor.tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hiddens[label] = [h[0, -1].float().cpu() for h in out.hidden_states]
        del model
        torch.cuda.empty_cache()

    if "ref" not in hiddens:
        report["layers"] = {"status": "no reference checkpoint"}
        return

    table = {}
    for label in ("fp8", "nvfp4"):
        if label not in hiddens:
            continue
        per_layer = [cos(r, q) for r, q in zip(hiddens["ref"], hiddens[label])]
        table[label] = per_layer
        drops = np.diff(per_layer)
        worst = int(np.argmin(drops)) + 1 if len(drops) else -1
        print(f"  {label:6s} layer 0 {per_layer[0]:.6f} -> "
              f"mid {per_layer[len(per_layer) // 2]:.6f} -> "
              f"final {per_layer[-1]:.6f}")
        print(f"         biggest single-layer drop at layer {worst} "
              f"({drops.min():.6f})" if len(drops) else "")
    report["layers"] = table
    print("\n  A smooth decline means accumulation and nothing local to exclude;")
    print("  a sharp knee names the layers worth excluding from NVFP4.")


def stage_channels(args, report: dict) -> None:
    """Is the final-layer error concentrated in a few high-magnitude channels?"""
    print("\n=== stage: channels — is the error carried by outliers? ===")

    if not args.nvfp4_ckpt:
        report["channels"] = {"status": "no nvfp4 checkpoint"}
        return

    prompt = ("You are an autonomous navigation assistant. Your task is to go to the "
              "kitchen. Where should you go next to stay on track?")
    hs = {}
    for label, path in (("ref", args.repkg_ckpt), ("nvfp4", args.nvfp4_ckpt)):
        model, processor, _ = _load(path, args.device)
        enc = processor.tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs[label] = out.hidden_states[-1][0, -1].float().cpu()
        del model
        torch.cuda.empty_cache()

    ref, q = hs["ref"], hs["nvfp4"]
    err = (q - ref).abs()
    order = torch.argsort(ref.abs(), descending=True)
    total = float((err ** 2).sum())

    result = {"baseline_cos": cos(ref, q), "share_by_topk": {}, "cos_masking_topk": {}}
    print(f"  baseline final-layer cosine: {result['baseline_cos']:.6f}")
    for k in (1, 4, 16, 36, 128):
        idx = order[:k]
        share = float((err[idx] ** 2).sum()) / max(total, 1e-30)
        mask = torch.ones_like(ref, dtype=torch.bool)
        mask[idx] = False
        result["share_by_topk"][k] = share
        result["cos_masking_topk"][k] = cos(ref[mask], q[mask])
        print(f"  top {k:4d} channels by |ref|: carry {100 * share:5.1f}% of squared error"
              f"  | cosine with them masked out: {result['cos_masking_topk'][k]:.6f}")

    report["channels"] = result
    best = max(result["cos_masking_topk"].values())
    if best > 0.99:
        print("\n  Masking a small number of channels restores the signal: this is outlier")
        print("  clipping under per-16 block scaling. AWQ or a targeted exclusion is the fix.")
    else:
        print("\n  The error is spread across channels, not carried by a few outliers.")
        print("  No amount of scaling or exclusion will recover it -- this is a capacity limit.")


def stage_bridge(args, report: dict) -> None:
    """End-to-end z_latents per checkpoint -- the number that decides."""
    print("\n=== stage: bridge — z_latents, the acceptance metric ===")
    from safetensors import safe_open

    bridge_path = os.path.join(args.repkg_ckpt, "bridge.safetensors")
    if not os.path.isfile(bridge_path):
        report["bridge"] = {"status": f"bridge.safetensors not found under {args.repkg_ckpt}"}
        print(f"  [skip] {report['bridge']['status']}")
        return
    with safe_open(bridge_path, framework="pt") as f:
        bt = {k: f.get_tensor(k) for k in f.keys()}
    cond = {k.replace("model.cond_projector.", ""): v.float().to(args.device)
            for k, v in bt.items() if "cond_projector" in k}

    prompt = ("You are an autonomous navigation assistant. Your task is to go to the "
              "kitchen. Where should you go next to stay on track?")
    z = {}
    for label, path in (("ref", args.repkg_ckpt), ("fp8", args.fp8_ckpt),
                        ("nvfp4", args.nvfp4_ckpt)):
        if not path:
            continue
        model, processor, _ = _load(path, args.device)
        enc = processor.tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -4:].float()
        x = torch.nn.functional.linear(h, cond["0.weight"], cond.get("0.bias"))
        x = torch.nn.functional.gelu(x)
        z[label] = torch.nn.functional.linear(x, cond["2.weight"], cond.get("2.bias")).cpu()
        del model
        torch.cuda.empty_cache()

    table = {}
    for label in ("fp8", "nvfp4"):
        if label in z:
            table[label] = cos(z["ref"], z[label])
            print(f"  {label:6s} z_latents cosine vs reference: {table[label]:.6f}")
    report["bridge"] = table

    if "nvfp4" in table:
        verdict = ("supported" if table["nvfp4"] >= 0.99
                   else "experimental" if table["nvfp4"] >= 0.95 else "rejected")
        print(f"\n  Verdict for NVFP4 on this model by the stated rule: {verdict}")
        report["verdict"] = verdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repkg_ckpt", required=True,
                   help="Repackaged System 2 checkpoint (the unquantized reference)")
    p.add_argument("--nvfp4_ckpt", default=None)
    p.add_argument("--fp8_ckpt", default=None)
    p.add_argument("--work_dir", default=os.path.expanduser("~/vln-opt-work"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", default="all",
                   choices=["all", "weights", "layers", "channels", "bridge"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = os.path.join(args.work_dir, "nvfp4_investigation.json")
    report = {}
    if os.path.isfile(out):
        with open(out) as f:
            report = json.load(f)

    stages = {"weights": stage_weights, "layers": stage_layers,
              "channels": stage_channels, "bridge": stage_bridge}
    todo = list(stages) if args.stage == "all" else [args.stage]
    for name in todo:
        stages[name](args, report)

    os.makedirs(args.work_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
