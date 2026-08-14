#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Capture real System-1 calibration tensors, for quantizing traj_dit and the memory block.

FP8 in TensorRT is explicit-quantization only, so the ONNX has to carry Q/DQ nodes, which
means a PTQ pass in PyTorch first, which means calibration data. For System 1 that data is
awkward to synthesize: ``traj_dit`` consumes ``z_latents`` produced by System 2 through
``cond_projector``, and the memory block consumes normalized navigation frames. Random
tensors have the wrong scale in both cases, and FP8 calibration is amax-based, so wrong
scale means wrong scale factors.

This runs the real thing. For each VLN sample it takes real frames, runs System 2's own
``generate_latents`` for a real ``z``, then taps every ``traj_dit`` call the diffusion loop
makes. The result is cached to disk because capturing it runs System 2 on every sample:
re-quantizing with a different config then costs a minute rather than the whole pipeline.

Run this under the transformers 4.51 environment (Python 3.10 here)::

    INTERNNAV_PATH=~/InternNav PYTHONPATH=~/InternNav \\
    python dump_system1_calib.py --calib_data_root work/calib_scenes \\
        --output_path work/system1_calib.pt
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "quantize"))

import internvla_compat  # noqa: E402

SEED = 12345


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--internvla_ckpt",
                   default=os.path.expanduser(
                       os.environ.get("INTERNVLA_CKPT",
                                      "~/InternNav/checkpoints/InternVLA-N1-DualVLN")))
    p.add_argument("--calib_data_root", required=True,
                   help="LeRobot scene root, as used by quantize/benchmark_accuracy.py")
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_samples", type=int, default=4,
                   help="VLN samples to draw. Each contributes num_inference_steps traj_dit "
                        "batches, so 4 gives 40 -- ample for amax calibration.")
    p.add_argument("--num_sample_trajs", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--num_frames", type=int, default=2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    internvla_compat.apply_all(need_system1=True, allow_missing_depth=False)

    import numpy as np
    import prompt_builder as pb
    from PIL import Image
    from benchmark_accuracy import discover_samples
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
    from transformers import AutoProcessor

    print(f"[1/4] Loading {args.internvla_ckpt}")
    config = InternVLAN1ModelConfig.from_pretrained(args.internvla_ckpt)
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.internvla_ckpt, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.internvla_ckpt,
                                              min_pixels=128 * 28 * 28,
                                              max_pixels=2048 * 32 * 32)

    print(f"[2/4] Drawing {args.num_samples} real VLN samples")
    samples = discover_samples(args.calib_data_root, args.num_samples, seed=SEED)
    if not samples:
        print(f"[ERROR] no samples under {args.calib_data_root}")
        return 1

    inner = model.get_model()
    dit_batches, image_batches = [], []

    _real_fwd = inner.traj_dit.forward

    def _tap(x, timestep, z_latents, *a, **kw):
        dit_batches.append((x.detach().half().cpu(),
                            timestep.detach().cpu(),
                            z_latents.detach().half().cpu()))
        return _real_fwd(x, timestep, z_latents, *a, **kw)

    print("[3/4] Running System 2 -> System 1 on each sample")
    torch.manual_seed(SEED)
    for i, sample in enumerate(samples):
        paths = sample["images"][-args.num_frames:]
        frames = [np.asarray(Image.open(p).convert("RGB").resize((224, 224))) / 255.0
                  for p in paths]
        images_dp = torch.from_numpy(np.stack(frames)).unsqueeze(0)
        images_dp = images_dp.to(args.device, torch.bfloat16)   # [1, T, 224, 224, 3]

        enc = pb.build_sample_inputs(sample, processor)
        enc = {k: v.to(args.device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            # The real bridge output, not a draw: generate_latents runs the LLM with the
            # learned latent queries appended and returns the normalized TRAJ hidden states.
            z = model.generate_latents(enc["input_ids"], enc.get("pixel_values"),
                                       enc.get("image_grid_thw"))

            chw = images_dp.permute(0, 1, 4, 2, 3)
            norm = ((chw - model._resnet_mean) / model._resnet_std).flatten(0, 1)
            image_batches.append(norm.to(torch.bfloat16).float().cpu())

            inner.traj_dit.forward = _tap
            model.generate_traj(z, images_dp,
                                num_sample_trajs=args.num_sample_trajs,
                                num_inference_steps=args.num_inference_steps)
            inner.traj_dit.forward = _real_fwd
        print(f"      sample {i + 1}/{len(samples)}: {len(dit_batches)} traj_dit batches")

    payload = {
        "dit_batches": dit_batches,
        "image_batches": image_batches,
        "num_sample_trajs": args.num_sample_trajs,
        "num_inference_steps": args.num_inference_steps,
    }
    print("[4/4] Writing")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"\nWrote {args.output_path} "
          f"({os.path.getsize(args.output_path) / 1e6:.1f} MB): "
          f"{len(dit_batches)} traj_dit batches, {len(image_batches)} image batches")
    print("Now run quantize_system1.py (same environment) to PTQ and build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
