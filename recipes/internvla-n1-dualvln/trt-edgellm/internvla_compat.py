# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime compatibility patches for loading InternVLA-N1 System 1.

These patches are applied in-memory (attribute reassignment) and never modify the InternNav
source tree. They fix three issues that otherwise prevent building the System 1 modules:

1. ``build_depthanythingv2`` loads the DepthAnything-V2 checkpoint from a relative path; the
   patch loads it from an absolute path (set via DAV2_CKPT / env), and can tolerate a missing
   file since ``from_pretrained`` overwrites ``model.rgb_model.*`` from safetensors afterwards.

2. ``LuminaNextDiT2DModel._set_gradient_checkpointing`` has an old signature that current
   diffusers calls with ``enable=`` / ``gradient_checkpointing_func=`` keywords; the patch
   accepts both.

3. ``build_traj_dit`` must pass ``ffn_dim_multiplier = 2/3`` (SwiGLU convention) so the traj_dit
   FFN shape matches the trained DualVLN checkpoint (1024, not 1536).
"""
from __future__ import annotations

import os

#: Root of the InternNav checkout to load the model from.
ACTIVE_INTERNNAV = os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))

#: DepthAnything-V2 checkpoint used by System 1's rgb_model.
DAV2_CKPT = os.path.expanduser(os.environ.get(
    "DAV2_CKPT", os.path.join(ACTIVE_INTERNNAV, "checkpoints/depth_anything_v2_vits.pth")))

#: FFN multiplier the DualVLN checkpoint was trained with (see patch_traj_dit_ffn).
TRAJ_DIT_FFN_MULTIPLIER = 2.0 / 3.0


def assert_active_tree() -> str:
    """Confirm ``internnav`` is imported from ACTIVE_INTERNNAV; fail early otherwise.

    Which tree loads depends on ``cwd`` (``''`` heads ``sys.path``), so a stray ``cd`` can
    silently swap the model code. This turns that silent failure into a loud one.
    """
    import internnav

    tree = os.path.dirname(os.path.dirname(internnav.__file__))
    if os.path.realpath(tree) != os.path.realpath(ACTIVE_INTERNNAV):
        raise RuntimeError(
            f"internnav loaded from the wrong tree:\n"
            f"  actual  : {tree}\n"
            f"  expected: {ACTIVE_INTERNNAV}\n"
            f"Usually the cwd sits inside a different InternNav checkout. Re-run elsewhere, "
            f"or put {ACTIVE_INTERNNAV} at the front of sys.path."
        )
    return tree


def patch_depth_anything(allow_missing: bool = False) -> bool:
    """Replace ``build_depthanythingv2`` with a version that loads from an absolute path.

    Only reassigns a module attribute in memory. Returns True once patched.
    """
    import torch

    import internnav.model.basemodel.internvla_n1.internvla_n1_arch as arch

    if getattr(arch, "_vlnopt_patched", False):
        return True

    if not os.path.isfile(DAV2_CKPT):
        if not allow_missing:
            raise FileNotFoundError(
                f"DepthAnything-V2 checkpoint not found: {DAV2_CKPT}\n"
                f"Required for System 1 (nextdit_async). Pass allow_missing=True to build the "
                f"architecture with random init — acceptable because from_pretrained overwrites "
                f"model.rgb_model.* from safetensors immediately afterwards."
            )
        print(f"[compat] WARNING: missing {DAV2_CKPT} — rgb_model randomly initialized, "
              f"relying on from_pretrained to load weights.")

    def _build_dav2_patched(config):
        from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import (
            DepthAnythingV2,
        )

        model_configs = {
            "vits": {"encoder": "vits", "features": 64,
                     "out_channels": [48, 96, 192, 384]}
        }
        dav2 = DepthAnythingV2(**model_configs["vits"])
        if os.path.isfile(DAV2_CKPT):
            dav2.load_state_dict(torch.load(DAV2_CKPT, map_location="cpu"))
        return dav2.pretrained

    arch.build_depthanythingv2 = _build_dav2_patched
    arch._vlnopt_patched = True
    return True


def patch_gradient_checkpointing() -> bool:
    """Reconcile the ``_set_gradient_checkpointing`` signature between nextdit and diffusers.

    nextdit defines ``(self, module, value)`` but current diffusers calls it with
    ``(enable=..., gradient_checkpointing_func=...)``. This fails while building traj_dit, so
    it must be patched before any System 1 build path.
    """
    try:
        from internnav.model.basemodel.internvla_n1.nextdit_traj import (
            LuminaNextDiT2DModel,
        )
    except ImportError:
        return False

    if getattr(LuminaNextDiT2DModel, "_vlnopt_gc_patched", False):
        return True

    def _set_gc_compat(self, module=None, value=False, enable=None,
                       gradient_checkpointing_func=None):
        v = enable if enable is not None else value
        if module is not None:
            module.gradient_checkpointing = v
        else:
            for m in self.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = v

    LuminaNextDiT2DModel._set_gradient_checkpointing = _set_gc_compat
    LuminaNextDiT2DModel._vlnopt_gc_patched = True
    return True


def patch_traj_dit_ffn(multiplier: float = TRAJ_DIT_FFN_MULTIPLIER) -> bool:
    """Patch ``build_traj_dit`` so the traj_dit FFN matches the checkpoint.

    The DualVLN checkpoint was trained with ``ffn_dim_multiplier = 2/3`` (SwiGLU convention),
    giving inner_dim 1024 for dim=384. The stock ``build_traj_dit`` never passes this, leaving
    inner_dim at 1536 and causing a state_dict size mismatch. Only feed_forward.linear_1/linear_3
    are affected; norm1.linear is 4*dim on both sides and already matches.
    """
    import internnav.model.basemodel.internvla_n1.internvla_n1_arch as arch

    if getattr(arch, "_vlnopt_ffn_patched", False):
        return True

    _orig = arch.build_traj_dit

    def _build_traj_dit_patched(config):
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import (
            NextDiTCrossAttn, NextDiTCrossAttnConfig,
        )

        dit_cfg = NextDiTCrossAttnConfig(
            latent_embedding_size=arch.LatentEmbSize,
            ffn_dim_multiplier=multiplier,   # the only difference from the stock builder
        )
        dit = NextDiTCrossAttn(dit_cfg)
        return dit, FlowMatchEulerDiscreteScheduler()

    _build_traj_dit_patched.__wrapped__ = _orig
    arch.build_traj_dit = _build_traj_dit_patched
    arch._vlnopt_ffn_patched = True
    return True


def apply_all(need_system1: bool = True, allow_missing_depth: bool = False) -> None:
    """Convenience entry point: verify the tree and apply the needed patches."""
    tree = assert_active_tree()
    print(f"[compat] internnav tree: {tree}")
    if need_system1:
        # Order matters: patch gradient-checkpointing first, since it fails inside the
        # traj_dit builder.
        if patch_gradient_checkpointing():
            print("[compat] patched LuminaNextDiT2DModel._set_gradient_checkpointing")
        patch_depth_anything(allow_missing=allow_missing_depth)
        print(f"[compat] patched build_depthanythingv2 -> {DAV2_CKPT}")
        patch_traj_dit_ffn()
        print(f"[compat] patched build_traj_dit -> ffn_dim_multiplier={TRAJ_DIT_FFN_MULTIPLIER:.4f}")
    # Needed on transformers 5.x regardless of System 1; harmless on 4.x.
    patch_config_flattening()


def patch_config_flattening() -> bool:
    """Re-expose the top-level LLM config fields that transformers 5.x nests.

    InternNav reads ``config.hidden_size``, ``config.num_hidden_layers`` and friends off
    the top-level config. transformers 4.x flattened them there; 5.x moves them under
    ``text_config``, so constructing the model raises a bare
    ``'InternVLAN1ModelConfig' object has no attribute 'hidden_size'``.

    This matters beyond tidiness: System-1 export needs InternNav (transformers 4.51) while
    the TensorRT Python bindings ship for 3.12 only, so any script needing *both* -- the
    System-1 parity check, for one -- cannot run without reconciling them. Copying the
    fields back from text_config is the smaller of the two evils; the alternative is
    pinning transformers 4.51 into the TensorRT environment and hoping the edgellm exporter
    still works there.
    """
    try:
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ModelConfig)
    except ImportError:
        return False

    _orig = InternVLAN1ModelConfig.from_pretrained.__func__

    def _from_pretrained(cls, *args, **kwargs):
        config = _orig(cls, *args, **kwargs)
        inner = getattr(config, "text_config", None)
        if inner is not None:
            for field in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                          "num_key_value_heads", "intermediate_size", "rms_norm_eps",
                          "vocab_size", "max_position_embeddings", "rope_theta"):
                if not hasattr(config, field) and hasattr(inner, field):
                    setattr(config, field, getattr(inner, field))
        return config

    InternVLAN1ModelConfig.from_pretrained = classmethod(_from_pretrained)
    print("[compat] re-exposed top-level LLM config fields (transformers 5.x nests them)")
    return True
