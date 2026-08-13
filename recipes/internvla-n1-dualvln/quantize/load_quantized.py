# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Load a ModelOpt-exported checkpoint back into PyTorch with its scales applied.

This exists because the obvious thing does not work. ``export_hf_checkpoint`` writes the
LLM weights as real ``torch.float8_e4m3fn`` with the dequantization scales in separate
``*.weight_scale`` / ``*.input_scale`` tensors. Calling
``AutoModelForImageTextToText.from_pretrained`` on that directory *appears* to succeed --
it warns that the scale tensors "were not used when initializing" and moves on -- but the
resulting model has every quantized weight off by its scale factor. Any accuracy measured
that way is measuring a broken load, not quantization error, and it will look far worse
than the engine actually is.

So reconstruct the weights explicitly::

    w_bf16 = w_fp8.to(bfloat16) * weight_scale

Two caveats worth stating plainly:

* This reproduces **weight** quantization error only. Real FP8 W8A8 also quantizes
  activations, which the TensorRT engine does and this does not. Weight error is the
  dominant term and this is the standard PyTorch-side proxy, but a number from here is a
  lower bound on the engine's deviation, not a prediction of it.
* Modules excluded from quantization (the vision tower, ``lm_head``) are stored in bf16
  already and pass through untouched, which is the intended behaviour.
"""
import glob
import json
import os
from typing import Optional

import torch
from safetensors import safe_open


# NVFP4 E2M1: 1 sign, 2 exponent, 1 mantissa bit. Sixteen representable values, two packed
# per stored byte, with a per-16-element FP8 block scale and one float32 global scale.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
NVFP4_BLOCK = 16


def unpack_nvfp4(packed: torch.Tensor, block_scale: torch.Tensor,
                 global_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct a bf16 weight from ModelOpt's packed NVFP4 representation.

    ``packed`` is uint8 of shape [out, in/2]: low nibble first, then high nibble.
    ``block_scale`` is FP8 of shape [out, in/16], one scale per 16 input elements.
    ``global_scale`` is a single float32 that rescales the whole tensor.
    """
    lut = _E2M1.to(packed.device)
    low = lut[(packed & 0x0F).long()]
    high = lut[(packed >> 4).long()]
    # Interleave back to the original width: low nibble is element 2i, high is 2i+1.
    out = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)

    scale = block_scale.to(torch.float32) * global_scale.to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK, dim=-1)
    if scale.shape[-1] != out.shape[-1]:
        raise ValueError(f"NVFP4 block scale expands to {scale.shape[-1]} columns but the "
                         f"unpacked weight has {out.shape[-1]}")
    return (out * scale).to(dtype)


def quant_algo(model_path: str) -> Optional[str]:
    """Return the quantization algorithm recorded by ModelOpt, or None if unquantized."""
    cfg = os.path.join(model_path, "hf_quant_config.json")
    if not os.path.isfile(cfg):
        return None
    with open(cfg) as f:
        return json.load(f).get("quantization", {}).get("quant_algo")


def _iter_shards(model_path: str):
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        if os.path.basename(shard) == "bridge.safetensors":
            continue
        yield shard


def dequantize_state_dict(model_path: str,
                          dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Read a ModelOpt checkpoint and return a plain state dict with scales folded in.

    Weights that were not quantized are returned as stored.
    """
    scales: dict[str, torch.Tensor] = {}
    raw: dict[str, torch.Tensor] = {}

    for shard in _iter_shards(model_path):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.endswith(("weight_scale", "input_scale", "weight_scale_2",
                                 "pre_quant_scale")):
                    scales[key] = tensor
                else:
                    raw[key] = tensor

    out: dict[str, torch.Tensor] = {}
    n_dequant = 0
    n_awq = 0
    for key, tensor in raw.items():
        # NVFP4: packed uint8 plus a block scale and a global scale.
        block = scales.get(key + "_scale")
        glob = scales.get(key + "_scale_2")
        if tensor.dtype == torch.uint8 and block is not None and glob is not None:
            w = unpack_nvfp4(tensor, block, glob, torch.float32)
            # AWQ-lite scales the activations by a per-input-channel s and stores the
            # weight pre-divided by it, so that y = (x * s) @ (W / s) reproduces x @ W.
            # A plain matmul needs s multiplied back in. Verified empirically on this
            # checkpoint against the unquantized weights, since the direction is easy to
            # get backwards: multiplying gives cosine 0.9898, leaving it alone 0.9606,
            # dividing 0.8089. Skipping this entirely reads as ~190% relative error and
            # looks like AWQ being catastrophically bad rather than loaded wrong.
            pqs = scales.get(key.replace(".weight", ".pre_quant_scale"))
            if pqs is not None:
                w = w * pqs.to(torch.float32)
                n_awq += 1
            out[key] = w.to(dtype)
            n_dequant += 1
            continue
        scale = scales.get(key + "_scale")
        if scale is not None and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out[key] = tensor.to(torch.float32).mul_(scale.to(torch.float32)).to(dtype)
            n_dequant += 1
        elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8 storage with no scale would silently corrupt the weight; refuse.
            raise ValueError(f"{key} is FP8 but has no matching {key}_scale in the checkpoint")
        else:
            out[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor

    awq_note = f", {n_awq} with an AWQ pre-quant scale folded out" if n_awq else ""
    print(f"      [load] dequantized {n_dequant} tensors{awq_note}, "
          f"{len(out) - n_dequant} passed through unchanged")
    return out


def load_for_eval(model_path: str, dtype: torch.dtype = torch.bfloat16,
                  device: str = "cuda"):
    """Load a checkpoint for evaluation, applying ModelOpt scales when present.

    Returns ``(model, processor, algo)`` where ``algo`` is the quantization algorithm
    string or None.
    """
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    algo = quant_algo(model_path)
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=128 * 28 * 28, max_pixels=2048 * 32 * 32)

    if algo is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    else:
        # Hand transformers the already-dequantized weights instead of letting it read the
        # checkpoint. Two reasons it cannot read this itself: FP8 tensors load without
        # their scales (silently wrong by 600-1800x), and NVFP4 tensors are packed two
        # values per byte, so from_pretrained fails outright on the halved width. Passing
        # state_dict= also avoids loading the whole checkpoint twice.
        state = dequantize_state_dict(model_path, dtype=dtype)
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            None, config=config, state_dict=state, torch_dtype=dtype,
            low_cpu_mem_usage=True)

    model = model.to(device=device, dtype=dtype).eval()
    return model, processor, algo
