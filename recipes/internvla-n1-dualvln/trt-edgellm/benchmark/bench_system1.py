#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark System 1 (generate_traj) latency in PyTorch and report Hz.
"""
import os
import sys
import time
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "lib"))
sys.path.append("/usr/lib/python3.12/dist-packages")  # cv2 (system) for depth_anything
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ACTIVE = os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT = os.path.join(ACTIVE, "checkpoints/InternVLA-N1-DualVLN")
REPKG = os.path.join(os.environ.get("WORK_DIR", os.path.expanduser("~/vln-opt-work")), "qwen25vl_system2")
IMG = os.path.expanduser("~/modelopt/TensorRT-Edge-LLM/examples/multimodal/pics/giant_panda.jpeg")


def timeit(fn, warm=2, n=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts), min(ts), max(ts)


def main():
    dev = "cuda"
    if ACTIVE not in sys.path:
        sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
    print("[1/3] Load full InternVLA (System1)")
    cfg = InternVLAN1ModelConfig.from_pretrained(CKPT)
    model = InternVLAN1ForCausalLM.from_pretrained(
        CKPT, config=cfg, torch_dtype=torch.bfloat16, attn_implementation=os.environ.get("ATTN", "sdpa"),
        low_cpu_mem_usage=True).to(dev).eval()

    print("[2/3] Prepare z_latents + images_dp (as the agent)")
    N_QUERY = model.get_n_query()
    # placeholder z_latents (cond_projector is applied inside generate_traj)
    traj_latents = torch.randn(1, N_QUERY, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    a = np.array(Image.open(IMG).convert("RGB").resize((224, 224))) / 255.0
    t = torch.from_numpy(a).float()
    images_dp = torch.stack([t, t]).unsqueeze(0).to(dev)   # [1,2,224,224,3]

    print("[3/3] Time generate_traj (System 1)\n" + "=" * 60)
    with torch.no_grad():
        for ns in (32, 4, 1):
            def fn(ns=ns):
                with torch.no_grad():
                    model.generate_traj(traj_latents.to(dev), images_dp,
                                        num_sample_trajs=ns, num_inference_steps=10)
            try:
                mean, mn, mx = timeit(fn, warm=2, n=5)
                print(
                    f"  num_sample_trajs={ns:>2}: {mean*1000:7.1f} ms  (min {mn*1000:.0f}, max {mx*1000:.0f})  → {1/mean:5.1f} Hz")  # noqa: E501
            except Exception as e:
                print(f"  num_sample_trajs={ns}: ERR {type(e).__name__}: {e}")
    print("\n  (paper target S1 = 30 Hz ≈ 33 ms; S2 = 2 Hz ≈ 500 ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
