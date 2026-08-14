#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Stage B of the System-1 parity check: run the engines on the reference's own inputs.

Reads the ``.pt`` written by ``dump_system1_reference.py`` and pushes the *same* tensors
through the memory and traj_dit engines, so the two halves never need to share an
interpreter — which they cannot, since InternNav wants transformers 4.x and the TensorRT
bindings are Python 3.12 only.

Two comparisons, deliberately separate:

* ``memory_tokens`` — the memory block alone (DepthAnythingV2 + MemoryEncoder + QFormer).
* the full trajectory — memory block plus the diffusion loop over traj_dit.

Checking the intermediate first means a mismatch localises to one engine instead of only
appearing at the end.

Run this under the TensorRT environment (Python 3.12 here)::

    EDGELLM_PLUGIN_PATH=... python verify/compare_system1_engines.py \\
        --reference_path work/system1_reference.pt --engine_dir work/onnx
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.append(os.environ.get("SYSTEM_SITE", "/usr/lib/python3.12/dist-packages"))

from trt_torch import Engine  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def out_of(result, key: str) -> torch.Tensor:
    if isinstance(result, dict):
        return result[key]
    return result[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_path", required=True)
    p.add_argument("--engine_dir", required=True,
                   help="Directory holding system1_traj_dit_bf16.engine and "
                        "system1_memory_bf16.engine")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--loop_dtype", default="bfloat16",
                   help="Host-side dtype for the diffusion loop. generate_traj runs it in "
                        "bfloat16; fp32 here diverges from the reference even with correct "
                        "engines, because the sampler amplifies the difference.")
    p.add_argument("--gate", type=float, default=0.99,
                   help="Minimum trajectory cosine to report PASS")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ref = torch.load(args.reference_path, map_location="cpu", weights_only=False)
    dev = args.device

    mem_path = os.path.join(args.engine_dir, "system1_memory_bf16.engine")
    dit_path = os.path.join(args.engine_dir, "system1_traj_dit_bf16.engine")
    for path in (mem_path, dit_path):
        if not os.path.isfile(path):
            print(f"[ERROR] engine not found: {path}")
            return 1

    print("[1/3] memory block")
    mem = Engine(mem_path)
    # The memory engine was built from MemBlock, which takes [T, C, H, W].
    images = ref["images_chw"].to(dev).float().contiguous()
    tokens = out_of(mem(images=images), "memory_tokens").float().cpu()
    mem.close()
    mem_cos = cos(ref["memory_tokens"], tokens)
    print(f"      memory_tokens cosine vs PyTorch: {mem_cos:.6f}")

    print("[2/3] traj_dit, single forward on the reference's own tensors")
    dit_probe = Engine(dit_path)
    st = ref["dit_step"]
    dit_probe.set_runtime_tensor_shape("z_latents", tuple(st["z_latents"].shape))
    one = out_of(dit_probe(x=st["x"].to(dev).contiguous(),
                           timestep=st["timestep"].to(dev).to(torch.int64).contiguous(),
                           z_latents=st["z_latents"].to(dev).contiguous()), "output")
    dit_probe.close()
    step_cos = cos(st["output"], one.float().cpu())
    print(f"      traj_dit single-step cosine: {step_cos:.6f}")

    print("[3/4] diffusion loop over traj_dit")
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    steps = int(ref["num_inference_steps"])
    n_traj = int(ref["num_sample_trajs"])
    dit = Engine(dit_path)

    enc_w = ref["action_encoder"]["weight"].to(dev)
    enc_b = ref["action_encoder"]["bias"].to(dev)
    dec_w = ref["action_decoder"]["weight"].to(dev)
    dec_b = ref["action_decoder"]["bias"].to(dev)

    # cond_projector is the System 2 -> System 1 bridge: Linear, GELU, Linear mapping the
    # 3584-wide hidden state down to the 768 traj_dit consumes. It stays on the host, so
    # apply it here from the weights the reference stage saved.
    cp = {k: v.to(dev) for k, v in ref["cond_projector"].items()}
    z = ref["z"].to(dev)
    z = torch.nn.functional.linear(z, cp["0.weight"], cp.get("0.bias"))
    z = torch.nn.functional.gelu(z)
    z = torch.nn.functional.linear(z, cp["2.weight"], cp.get("2.bias"))
    # generate_traj runs classifier-free guidance: the conditioning is [null, real] and the
    # latents are duplicated, so the engine's batch is 2 * num_sample_trajs. That is why the
    # traj_dit engine was built at 64 rather than 32.
    hidden = torch.cat([tokens.to(dev), z], dim=1)                    # [1, 36, 768]
    cond = torch.cat([torch.zeros_like(hidden), hidden], 0)           # [2, 36, 768]
    cond = cond.repeat_interleave(n_traj, dim=0).contiguous()         # [2*ns, 36, 768]
    pos_embed = ref["pos_embed"].to(dev)

    # Does the loop's own conditioning match what the reference actually fed traj_dit? If
    # not, a trajectory mismatch is this reimplementation's, not the engine's.
    ref_cond = ref["dit_step"]["z_latents"]
    print(f"      z_latents vs the reference's own: {cos(ref_cond, cond.float().cpu()):.6f}")

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(steps, sigmas=np.linspace(1.0, 1 / steps, steps))

    # The reference's own starting noise, captured at stage A. Redrawing it here would give
    # a different valid trajectory and look like an engine failure.
    latents = ref["init_latents"].to(dev)
    dt = getattr(torch, args.loop_dtype)
    latents = latents.to(dt)
    cond, pos_embed = cond.to(dt), pos_embed.to(dt)
    enc_w, enc_b, dec_w, dec_b = (t.to(dt) for t in (enc_w, enc_b, dec_w, dec_b))
    dit.set_runtime_tensor_shape("z_latents", tuple(cond.shape))
    for t in scheduler.timesteps:
        feats = torch.nn.functional.linear(latents, enc_w, enc_b) + pos_embed
        feats = feats.repeat(2, 1, 1)
        if hasattr(scheduler, "scale_model_input"):
            feats = scheduler.scale_model_input(feats, t)
        ts = t.to(dev).expand(feats.shape[0]).to(torch.int64).contiguous()
        pred = out_of(dit(x=feats.float().contiguous(), timestep=ts,
                          z_latents=cond.float().contiguous()), "output").to(dt)
        pred = torch.nn.functional.linear(pred, dec_w, dec_b)
        uncond, condit = pred.chunk(2)
        pred = uncond + args.guidance_scale * (condit - uncond)
        latents = scheduler.step(pred, t, latents).prev_sample
    dit.close()
    traj = latents.float().cpu()

    print("[4/4] compare")
    ref_traj = ref["trajectory"]
    if traj.shape != ref_traj.shape:
        print(f"      shape mismatch: engine {tuple(traj.shape)} vs "
              f"reference {tuple(ref_traj.shape)}")
        print("      The diffusion loop here reimplements generate_traj; if the shapes")
        print("      disagree, the reimplementation is wrong, not the engines.")
        return 1

    traj_cos = cos(ref_traj, traj)
    l2 = float((traj - ref_traj).norm() / ref_traj.norm())
    print("=" * 58)
    print(f"  memory_tokens cosine : {mem_cos:.6f}")
    print(f"  traj_dit single step : {step_cos:.6f}")
    print(f"  trajectory cosine    : {traj_cos:.6f}")
    print(f"  trajectory rel-L2    : {l2:.4f}")
    ok = traj_cos >= args.gate
    print(f"  {'PASS' if ok else 'BELOW GATE'} (gate {args.gate})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
