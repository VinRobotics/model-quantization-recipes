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
                if key.endswith(("weight_scale", "input_scale", "weight_scale_2")):
                    scales[key] = tensor
                else:
                    raw[key] = tensor

    out: dict[str, torch.Tensor] = {}
    n_dequant = 0
    for key, tensor in raw.items():
        scale = scales.get(key + "_scale")
        if scale is not None and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out[key] = tensor.to(torch.float32).mul_(scale.to(torch.float32)).to(dtype)
            n_dequant += 1
        elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8 storage with no scale would silently corrupt the weight; refuse.
            raise ValueError(f"{key} is FP8 but has no matching {key}_scale in the checkpoint")
        else:
            out[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor

    print(f"      [load] dequantized {n_dequant} FP8 tensors, "
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

    # from_pretrained gives a correctly wired model with its buffers (rope inv_freq and
    # friends) materialised. For a quantized checkpoint it silently drops the scale
    # tensors and casts the FP8 weights straight to bf16, so those weights are wrong by
    # their scale factor -- they get overwritten below. Building on a meta device instead
    # would avoid the wasted load but leaves the buffers unmaterialised.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    if algo is not None:
        state = dequantize_state_dict(model_path, dtype=dtype)
        own = dict(model.named_parameters())
        own.update(dict(model.named_buffers()))
        n_fixed = 0
        with torch.no_grad():
            for key, tensor in state.items():
                target = own.get(key)
                if target is None:
                    continue
                if target.shape != tensor.shape:
                    raise RuntimeError(f"shape mismatch for {key}: "
                                       f"model {tuple(target.shape)} vs "
                                       f"checkpoint {tuple(tensor.shape)}")
                target.copy_(tensor.to(target.dtype))
                n_fixed += 1
        print(f"      [load] applied {n_fixed} dequantized tensors over the raw load")

    model = model.to(device=device, dtype=dtype).eval()
    return model, processor, algo
