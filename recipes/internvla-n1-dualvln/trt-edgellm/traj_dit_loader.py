#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Load the InternVLA-N1 System 1 traj_dit (NextDiT) core, and export it to ONNX.

The traj_dit is exported in FP32 (its RoPE freqs_cis are float32; FP32 export avoids
mixed-precision trace errors). Precision is set to BF16 at engine-build time. The surrounding
host-side ops (action encoder/decoder, cond_projector, pos_encoding, CFG blend, scheduler)
stay in PyTorch. Legacy exporter (dynamo=False, opset 19).

Export shapes:
    input : x [2N, 32, 384] · timestep [2N] int64 · z_latents [2N, Z, 768]
    output: noise_pred [2N, 32, 384]
"""
import os
import sys

import torch

_LIB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LIB)
sys.path.insert(0, os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav")))

MODEL = os.environ.get("INTERNVLA_CKPT", os.path.join(
    os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav")),
    "checkpoints/InternVLA-N1-DualVLN"))
OUT = os.path.join(os.environ.get("VLN_OPT_WORK", os.path.expanduser("~/vln-opt-work")),
                   "onnx/system1_traj_dit.onnx")


def load_traj_dit():
    """Return (traj_dit module, LatentEmbSize). The rgb_model (DepthAnything) is stubbed
    out since only the traj_dit is needed here."""
    from internvla_compat import patch_gradient_checkpointing, patch_traj_dit_ffn

    patch_gradient_checkpointing()
    patch_traj_dit_ffn()  # requires diffusers 0.33.1 -> inner_dim 1024
    import internnav.model.basemodel.internvla_n1.internvla_n1_arch as _arch

    class _DepthStub(torch.nn.Module):
        def forward(self, *a, **k):
            raise RuntimeError("rgb_model stub")

    _arch.build_depthanythingv2 = lambda config: _DepthStub()

    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig,
    )

    config = InternVLAN1ModelConfig.from_pretrained(MODEL)
    model = InternVLAN1ForCausalLM.from_pretrained(
        MODEL, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to("cuda").eval()
    return model.get_model().traj_dit, _arch.LatentEmbSize


class TrajDiTWrapper(torch.nn.Module):
    """Adapt positional args to keyword call (torch.onnx.export passes positionally)."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, timestep, z_latents):
        return self.dit(x=x, timestep=timestep, z_latents=z_latents)


def main():
    N, WP, DIM = 32, 32, 384          # num_sample_trajs, waypoints, dim
    dev = "cuda"
    print("[1/3] Load traj_dit (FP32)")
    dit, zdim = load_traj_dit()
    dit = dit.to(torch.float32).eval()
    print(f"      z_latents dim (LatentEmbSize) = {zdim}")

    x = torch.randn(2 * N, WP, DIM, dtype=torch.float32, device=dev)
    timestep = torch.ones(2 * N, dtype=torch.int64, device=dev)
    z_latents = torch.randn(2 * N, 4, zdim, dtype=torch.float32, device=dev)
    wrapped = TrajDiTWrapper(dit).eval()

    with torch.no_grad():
        ref = wrapped(x, timestep, z_latents)
    print(f"[2/3] Forward OK -> output {tuple(ref.shape)} (expected [{2*N},{WP},{DIM}])")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # batch (2N) dynamic for flexible num_sample_trajs; seq (32) & n_query (4) static
    dyn = {"x": {0: "batch"}, "timestep": {0: "batch"},
           "z_latents": {0: "batch"}, "output": {0: "batch"}}
    print(f"[3/3] Export ONNX (dynamo=False, opset 19) -> {OUT}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapped, (x, timestep, z_latents), OUT,
            input_names=["x", "timestep", "z_latents"],
            output_names=["output"],
            opset_version=19, do_constant_folding=True, export_params=True,
            dynamic_axes=dyn, dynamo=False)
    print(f"      Done. {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(OUT))
        print("      onnx.checker: valid")
    except Exception as e:
        print(f"      onnx.checker error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
