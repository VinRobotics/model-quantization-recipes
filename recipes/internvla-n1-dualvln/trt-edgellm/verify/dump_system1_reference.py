#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Stage A of the System-1 parity check: capture the PyTorch reference to disk.

The check cannot run in one interpreter. InternNav targets transformers 4.x, while the
TensorRT Python bindings ship for Python 3.12 where transformers is 5.x — under which
InternNav fails on `config.hidden_size`, then on `apply_chunking_to_forward`, and so on
without a natural end.

Splitting it across the two environments avoids that entirely: this stage runs the PyTorch
reference where InternNav works and writes its inputs and outputs to a `.pt` file;
`compare_system1_engines.py` reads that file where TensorRT works. Nothing needs both at
once, and the comparison stays exact because the *same* tensors cross the boundary — the
engines are fed the reference's own inputs rather than regenerated ones.

Run this under the transformers 4.51 environment (Python 3.10 here)::

    INTERNNAV_PATH=~/InternNav INTERNVLA_CKPT=~/InternNav/checkpoints/InternVLA-N1-DualVLN \\
    python verify/dump_system1_reference.py --output_path work/system1_reference.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import internvla_compat  # noqa: E402

SEED = 12345


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--internvla_ckpt",
                   default=os.path.expanduser(
                       os.environ.get("INTERNVLA_CKPT",
                                      "~/InternNav/checkpoints/InternVLA-N1-DualVLN")))
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_sample_trajs", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    internvla_compat.apply_all(need_system1=True, allow_missing_depth=False)

    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)

    print(f"[1/3] Loading {args.internvla_ckpt}")
    config = InternVLAN1ModelConfig.from_pretrained(args.internvla_ckpt)
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.internvla_ckpt, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device).eval()

    print("[2/3] Building fixed inputs")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    inner = model.get_model()
    z_dim = inner.latent_queries.shape[-1]
    # The bridge output System 1 consumes. Random but seeded, so the engine stage sees
    # exactly these numbers rather than its own draw.
    z = torch.randn(1, 4, z_dim, dtype=torch.bfloat16, device=args.device)
    # generate_traj permutes (0, 1, 4, 2, 3), so it wants [B, T, H, W, C] and produces
    # [B, T, C, H, W] internally. The memory engine, built from MemBlock, takes the
    # already-permuted [T, C, H, W] -- both forms are saved so stage B feeds each the
    # layout it expects.
    images_dp = torch.randn(1, args.num_frames, 224, 224, 3,
                            dtype=torch.bfloat16, device=args.device)

    # generate_traj draws its starting noise with randn_tensor part-way through the
    # function, after forwards that may or may not touch the global RNG. Reseeding in stage B
    # therefore does not reproduce it -- and a different draw yields a different but equally
    # valid trajectory, which reads as a broken engine (cosine ~0.3). Capture the actual
    # tensor instead of trying to redraw it.
    from internnav.model.basemodel.internvla_n1 import internvla_n1 as _n1
    _real_randn = _n1.randn_tensor
    captured = {}

    def _capture(*a, **kw):
        out = _real_randn(*a, **kw)
        captured.setdefault("latents", out.detach().float().cpu())
        return out

    _n1.randn_tensor = _capture

    # Capture one real traj_dit call, inputs and output together. The diffusion loop in
    # stage B is a reimplementation, so a trajectory mismatch on its own cannot tell a bad
    # engine from a bad reimplementation. A single forward on the module's own tensors can.
    _dit = inner.traj_dit
    _real_fwd = _dit.forward
    step = {}

    def _tap(x, timestep, z_latents, *a, **kw):
        out = _real_fwd(x, timestep, z_latents, *a, **kw)
        step.setdefault("x", x.detach().float().cpu())
        step.setdefault("timestep", timestep.detach().cpu())
        step.setdefault("z_latents", z_latents.detach().float().cpu())
        step.setdefault("output", out.detach().float().cpu())
        return out

    _dit.forward = _tap

    print("[3/3] Reference generate_traj")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with torch.no_grad():
        traj = model.generate_traj(z, images_dp,
                                   num_sample_trajs=args.num_sample_trajs,
                                   num_inference_steps=args.num_inference_steps)
    _n1.randn_tensor = _real_randn
    _dit.forward = _real_fwd
    traj = traj.float().cpu()
    if "latents" not in captured:
        print("[ERROR] randn_tensor was never called -- the capture hook missed the draw")
        return 1
    print(f"      trajectory {tuple(traj.shape)}")

    # The intermediate the engines replace, so a mismatch can be localised to the memory
    # block or to the diffusion head rather than only showing up at the end.
    with torch.no_grad():
        # generate_traj normalizes with the ResNet statistics before the memory block, and
        # MemBlock -- the module that was exported to ONNX -- does not, so it expects the
        # already-normalized tensor. Feeding raw pixels here makes memory_tokens disagree
        # with what generate_traj actually used (measured 0.315) while both halves look
        # internally consistent.
        chw = images_dp.permute(0, 1, 4, 2, 3)
        # The statistics are fp32 buffers, so the division promotes; generate_traj casts
        # back with .to(dtype) before rgb_model and this must match.
        images_chw = ((chw - model._resnet_mean) / model._resnet_std)
        images_chw = images_chw.flatten(0, 1).to(torch.bfloat16)
        # Use MemBlock itself rather than re-deriving the sequence: it is exactly what was
        # exported to ONNX, so any mismatch downstream is the engine's, not a difference in
        # how the reference was computed.
        from memblock import MemBlock
        block = MemBlock(inner.rgb_model, inner.memory_encoder,
                         inner.rgb_resampler).to(args.device).eval()
        memory_tokens = block(images_chw).float().cpu()
        # SinusoidalPositionalEncoding stays on the host, and generate_traj adds it to the
        # action features every diffusion step. Dump the evaluated tensor rather than its
        # weights so stage B does not have to reimplement the encoding.
        wp = traj.shape[-2]
        pos_ids = torch.arange(wp, device=args.device).reshape(1, -1)
        pos_embed = inner.pos_encoding(pos_ids).float().cpu()

    payload = {
        "seed": SEED,
        "z": z.float().cpu(),
        "images_dp": images_dp.float().cpu(),          # [B, T, H, W, C], generate_traj form
        "images_chw": images_chw.float().cpu(),        # [T, C, H, W], memory-engine form
        "memory_tokens": memory_tokens,
        "pos_embed": pos_embed,
        "init_latents": captured["latents"],
        "dit_step": step,
        "trajectory": traj,
        "num_sample_trajs": args.num_sample_trajs,
        "num_inference_steps": args.num_inference_steps,
        "cond_projector": {k: v.float().cpu()
                           for k, v in inner.cond_projector.state_dict().items()},
        "action_encoder": {k: v.float().cpu()
                           for k, v in inner.action_encoder.state_dict().items()},
        "action_decoder": {k: v.float().cpu()
                           for k, v in inner.action_decoder.state_dict().items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(payload, args.output_path)
    size_mb = os.path.getsize(args.output_path) / 1e6
    print(f"\nWrote {args.output_path} ({size_mb:.1f} MB)")
    print("Now run verify/compare_system1_engines.py under the TensorRT environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
