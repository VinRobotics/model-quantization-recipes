#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Measure peak GPU memory (torch.cuda.max_memory_allocated) of the full PyTorch
pipeline (System 2 weights + System 1 forward).
"""
import os
import sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "lib"))
sys.path.append("/usr/lib/python3.12/dist-packages")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ACTIVE = os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT = os.path.join(ACTIVE, "checkpoints/InternVLA-N1-DualVLN")
IMG = os.path.expanduser("~/modelopt/TensorRT-Edge-LLM/examples/multimodal/pics/giant_panda.jpeg")
GB = 1024**3


def peak(): return torch.cuda.max_memory_allocated() / GB  # noqa: E704


def reset():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def main():
    dev = "cuda"
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    if ACTIVE not in sys.path:
        sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
    reset()
    print("[1] Load full InternVLA (PyTorch, System1+System2)")
    cfg = InternVLAN1ModelConfig.from_pretrained(CKPT)
    model = InternVLAN1ForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa", low_cpu_mem_usage=True).to(dev).eval()
    w_mem = torch.cuda.memory_allocated() / GB
    print(f"    weights loaded (S2+S1): {w_mem:.2f} GB")

    NQ = model.get_n_query()
    z = torch.randn(1, NQ, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    a = np.array(Image.open(IMG).convert("RGB").resize((224, 224))) / 255.0
    tt = torch.from_numpy(a).float()
    images_dp = torch.stack([tt, tt]).unsqueeze(0).to(dev)

    print("[2] S1 generate_traj peak (PyTorch)")
    reset()
    with torch.no_grad():
        model.generate_traj(z, images_dp, num_sample_trajs=32, num_inference_steps=10)
    torch.cuda.synchronize()
    print(f"    S1 peak (incl. weights): {peak():.2f} GB")

    print("\n=== PyTorch peak GPU mem ===")
    print(f"  weights S2+S1 : {w_mem:.2f} GB")
    print(f"  peak during S1: {peak():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
