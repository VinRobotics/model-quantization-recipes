#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
quantize.py — PTQ for Qwen3.8-27B (model_type "qwen3_5") via llmcompressor.

Strategies
----------
  fp8    W8A8-FP8   — per-channel weight scales, activation scales from calibration
  nvfp4  W4A4-NVFP4 — NVIDIA FP4 with FP8 block scales + FP32 global scale

Scope
-----
Only the *language model* is quantized. The vision tower, the SSM recurrence
inputs of the Gated DeltaNet layers, every normalization, the MTP head and
lm_head stay in bf16. The ignore list is derived from the live module tree at
runtime (see inspect_model.py, which shares the same policy table), so it
follows the architecture instead of assuming layer indices.

Calibration data is CNN/DailyMail (text-only) — the language model is the only
thing being calibrated, so text-only inputs are the right signal and they skip
the VLM processor entirely.

Checkpoint layout follows the Cosmos-Reason2 convention: save_pretrained writes
the quantized safetensors + config.json, then every other file (tokenizer,
preprocessor, chat template, generation config, ...) is copied verbatim from
the base checkpoint.

Usage:
    python quantize.py --model_path .../Qwen3.8-27B \
                       --output_path .../outputs/qwen38-27b-fp8 \
                       --strategy fp8
"""

import argparse
import inspect
import re
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

import transformers
from transformers import AutoConfig, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Single source of truth for the "what do we quantize" policy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_model import _POLICY, classify, summarize_ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quantize")


# ---------------------------------------------------------------------------
# Strategy -> llmcompressor scheme
# ---------------------------------------------------------------------------

# (scheme, needs_calibration, min_compute_capability, note)
_STRATEGIES = {
    "fp8": (
        "FP8", True, 8.9,
        "W8A8 FP8, static per-tensor activation scales fitted on the calibration set",
    ),
    "fp8-dynamic": (
        "FP8_DYNAMIC", False, 8.9,
        "W8A8 FP8, per-channel weights + per-token dynamic activation scales — "
        "data-free, usually the most accurate FP8 variant, but ignores --dataset_id",
    ),
    "fp8-block": (
        "FP8_BLOCK", False, 8.9,
        "W8A8 FP8 with 128x128 block-wise weight scales and group-128 dynamic "
        "activations — matches the official Qwen3.5-family FP8 config "
        "(weight_block_size [128,128], activation_scheme dynamic)",
    ),
    "nvfp4": (
        "NVFP4", True, 10.0,
        "W4A4 NVFP4 with FP8 per-block scales — needs Blackwell (SM100+) to run fast",
    ),
    "nvfp4a16": (
        "NVFP4A16", True, 8.0,
        "NVFP4 weights with bf16 activations — runs anywhere, weight-only compression",
    ),
}


# ---------------------------------------------------------------------------
# SmoothQuant mappings
#
# Same two-mapping structure as the Cosmos-Reason2 pipeline, retargeted to this
# architecture. SmoothQuant divides the balance layer's output scale and
# multiplies it back into the consuming Linears, so the product is preserved
# exactly — but ONLY for the consumers listed here. Any consumer of the same
# norm that is left out silently receives mis-scaled activations, so both
# branches of the hybrid stack have to be enumerated:
#
#   input_layernorm  -> full_attention layers : self_attn.{q,k,v}_proj
#                    -> linear_attention      : linear_attn.in_proj_{qkv,z,a,b}
#   post_attention_layernorm -> mlp.{gate,up}_proj
#
# in_proj_a / in_proj_b are listed even though they stay bf16: they still
# consume the rescaled norm output and must absorb the compensating factor.
#
# The vision tower uses norm1 / norm2 (LayerNorm), so these patterns cannot
# reach it — the same non-overlap the Cosmos pipeline relies on.
#
# Cosmos writes its mappings as global regexes, which worked under llmcompressor
# 0.10. Under 0.13 `match_modules_set` groups a mapping's patterns into sets and
# rejects any set holding more than one smooth layer. A global regex over a
# *hybrid* stack breaks that: `self_attn.q_proj` exists in only 16 of the 64
# layers, so the 48 linear-attention `input_layernorm`s pile up unmatched until
# a full-attention layer finally closes the set, and resolution fails with
# "must match a single smooth layer".
#
# So the mappings are built per decoder layer from the live module tree, using
# exact module names. Each set then contains exactly one norm and precisely the
# Linears that consume it, whichever branch the layer happens to be.

_ATTN_CONSUMERS = {
    # full_attention layers
    "self_attn":   ("q_proj", "k_proj", "v_proj"),
    # linear_attention layers (Gated DeltaNet). in_proj_a / in_proj_b are listed
    # even though they stay bf16 — they still consume the rescaled norm output
    # and must absorb the compensating factor or their input is silently wrong.
    "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
}


def build_smoothquant_mappings(model) -> list:
    """One mapping per (decoder layer, norm) pair, using exact module names."""
    names = set(dict(model.named_modules()))
    mappings = []
    n_full = n_linear = 0

    for layer_name, layer in model.named_modules():
        if not re.search(r"\.layers\.\d+$", layer_name):
            continue
        if "visual" in layer_name or "vision" in layer_name:
            continue
        children = dict(layer.named_children())

        for branch, projections in _ATTN_CONSUMERS.items():
            if branch not in children:
                continue
            norm = f"{layer_name}.input_layernorm"
            balance = [f"{layer_name}.{branch}.{p}" for p in projections
                       if f"{layer_name}.{branch}.{p}" in names]
            if balance and norm in names:
                mappings.append([balance, norm])
                n_full += branch == "self_attn"
                n_linear += branch == "linear_attn"

        norm = f"{layer_name}.post_attention_layernorm"
        balance = [f"{layer_name}.mlp.{p}" for p in ("gate_proj", "up_proj")
                   if f"{layer_name}.mlp.{p}" in names]
        if balance and norm in names:
            mappings.append([balance, norm])

    logger.info("SmoothQuant mappings: %d total (%d full-attention, %d linear-attention, "
                "%d mlp)", len(mappings), n_full, n_linear, len(mappings) - n_full - n_linear)
    return mappings


# ---------------------------------------------------------------------------
# Fused-layer global scales (NVFP4 only)
# ---------------------------------------------------------------------------

def patch_fused_layer_names() -> None:
    """Teach llmcompressor that the Gated DeltaNet in-projections are fused.

    NVFP4 carries one FP32 *global* scale per tensor. vLLM packs several
    checkpoint modules into a single kernel, and every module inside a pack has
    to share that global scale or the fused GEMM cannot represent them. vLLM's
    packed_modules_mapping for qwen3_5 is:

        in_proj_qkvz = [in_proj_qkv, in_proj_z]
        in_proj_ba   = [in_proj_b,   in_proj_a]

    llmcompressor's FUSED_LAYER_NAMES covers the usual (q,k,v) and
    (gate,up) groups — both of which this model has and which come out
    correctly fused — but knows nothing about the hybrid stack's in_proj_qkv /
    in_proj_z pair. Left alone they get independent global scales (measured:
    328.0 vs 472.0 on layer 0), and vLLM warns at load:

        In NVFP4 linear, the weight global scale is different for parallel
        layers ... This will likely result in reduced accuracy.

    That is 96 modules across the 48 linear-attention layers — 15.4% of all
    Linear weight — running through a mis-scaled fused GEMM.

    (in_proj_b / in_proj_a are deliberately held at bf16, so that pack is
    internally consistent and must NOT be added here: fuse_weight_observers
    asserts that every layer in a group has a weight observer.)
    """
    from llmcompressor.observers import helpers

    pair = ("in_proj_qkv", "in_proj_z")
    if pair not in helpers.FUSED_LAYER_NAMES:
        helpers.FUSED_LAYER_NAMES.append(pair)
        logger.info("Patched FUSED_LAYER_NAMES with %s so vLLM's in_proj_qkvz pack "
                    "shares one NVFP4 global scale", pair)


def detect_compute_capability() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return min(
        major + minor / 10
        for major, minor in (torch.cuda.get_device_capability(i)
                             for i in range(torch.cuda.device_count()))
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_model_class(config):
    """Pick the concrete transformers class named in config.architectures.

    Qwen3.8-27B is a `Qwen3_5ForConditionalGeneration`; AutoModelForCausalLM does
    not necessarily map conditional-generation VLM heads, so resolve by name and
    only fall back to the Auto classes.
    """
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            logger.info("Resolved model class from config.architectures: %s", arch)
            return cls
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, auto_name, None)
        if cls is not None:
            logger.warning("Falling back to %s", auto_name)
            return cls
    raise RuntimeError("Could not resolve a model class for this checkpoint")


def load_model(model_path: str, device_map: str, max_memory):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls = resolve_model_class(config)
    logger.info("Loading weights from %s ...", model_path)
    model = cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()

    # llmcompressor reads attention metadata off the top-level config; on this
    # checkpoint it lives under text_config. Mirror it up so the KV/attention
    # bookkeeping resolves. Same fix the Cosmos-Reason2 pipeline needed.
    tc = getattr(config, "text_config", None)
    if tc is not None:
        for key in ("num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size"):
            if hasattr(tc, key) and not hasattr(model.config, key):
                setattr(model.config, key, getattr(tc, key))

    return model, config


def detect_sequential_targets(model, config) -> list:
    """Find the decoder-layer class names to use as llmcompressor sequential targets.

    Detected by counting: the classes that appear exactly num_hidden_layers times
    inside the text stack are the decoder layers. Hardcoding a class name breaks
    the moment the architecture is renamed upstream.
    """
    tc = getattr(config, "text_config", config)
    n_layers = getattr(tc, "num_hidden_layers", None)
    counts = Counter()
    for name, module in model.named_modules():
        if ".layers." in name and name.count(".layers.") == 1:
            tail = name.split(".layers.")[1]
            if tail.isdigit() and "visual" not in name and "vision" not in name:
                counts[type(module).__name__] += 1

    targets = [cls for cls, c in counts.items() if n_layers is None or c == n_layers]
    if not targets and counts:
        targets = [counts.most_common(1)[0][0]]
    logger.info("Sequential targets: %s  (layer-class counts: %s)", targets, dict(counts))
    return targets


# ---------------------------------------------------------------------------
# Ignore list
# ---------------------------------------------------------------------------

def build_ignore_list(model, extra: list) -> list:
    """Every nn.Linear whose role policy says keep_bf16, plus user overrides."""
    keep = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and _POLICY[classify(name)][0] == "keep_bf16"
    ]
    quantized = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and _POLICY[classify(name)][0] == "quantize"
    ]

    by_role = Counter(classify(n) for n in keep)
    logger.info("Holding %d Linear modules at bf16 %s", len(keep), dict(by_role))
    logger.info("Quantizing %d Linear modules %s",
                len(quantized), dict(Counter(classify(n) for n in quantized)))
    logger.info("Ignore patterns (collapsed): %s", summarize_ignore(keep))

    return sorted(set(keep) | set(extra))


# ---------------------------------------------------------------------------
# Calibration data — CNN/DailyMail, text-only
# ---------------------------------------------------------------------------

def build_calibration_dataset(tokenizer, dataset_id, dataset_config, dataset_split,
                              text_field, num_samples, max_seq_len, seed) -> Dataset:
    logger.info("Streaming calibration set %s/%s split=%s",
                dataset_id, dataset_config or "-", dataset_split)
    kwargs = dict(split=dataset_split, streaming=True)
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(dataset_id, **kwargs)
    # buffer_size 3x the sample count: enough shuffle entropy without holding the
    # whole stream in memory.
    ds = ds.shuffle(seed=seed, buffer_size=num_samples * 3).take(num_samples)

    fields = [f.strip() for f in text_field.split(",") if f.strip()]
    samples, last = [], None
    for item in ds:
        last = item
        text = next((str(item[f]).strip() for f in fields if item.get(f)), None)
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=max_seq_len, return_tensors="pt")
        samples.append({
            "input_ids": enc["input_ids"][0].tolist(),
            "attention_mask": enc["attention_mask"][0].tolist(),
        })

    if not samples:
        raise ValueError(
            f"No samples tokenized. Tried fields {fields}; "
            f"available: {list(last.keys()) if last else '<empty stream>'}"
        )
    lengths = [len(s["input_ids"]) for s in samples]
    logger.info("Calibration: %d samples, token length min/mean/max = %d/%d/%d",
                len(samples), min(lengths), sum(lengths) // len(lengths), max(lengths))
    return Dataset.from_list(samples)


def data_collator(batch):
    assert len(batch) == 1, "Calibration runs at batch_size=1"
    return {k: torch.tensor(v).unsqueeze(0) for k, v in batch[0].items()}


# ---------------------------------------------------------------------------
# Save — Cosmos-Reason2 convention
# ---------------------------------------------------------------------------

def postprocess_config(config_path: Path) -> None:
    """Strip keys some runtimes reject in the compressed-tensors block."""
    def drop(obj, keys):
        if isinstance(obj, dict):
            return {k: drop(v, keys) for k, v in obj.items() if k not in keys}
        if isinstance(obj, list):
            return [drop(i, keys) for i in obj]
        return obj

    with open(config_path) as f:
        cfg = json.load(f)
    cfg = drop(cfg, {"zp_dtype", "scale_dtype"})
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Post-processed %s", config_path)


def save_checkpoint(model, model_path: Path, output_dir: Path) -> None:
    """Write quantized weights, then copy every other base-model file verbatim.

    Deliberately does NOT call processor.save_pretrained(): re-serializing the
    processor rewrites preprocessor_config.json with whatever the calibration run
    left on it, which breaks inference. Copying the originals is what the
    Cosmos-Reason2 pipeline does and it is the safe move.
    """
    logger.info("Saving quantized weights -> %s", output_dir)
    # save_original_format=False is required here. llmcompressor's sequential
    # pipeline leaves modules CPU-offloaded, and transformers >=5.x otherwise
    # tries to revert its weight conversions at save time — which needs every
    # tensor of a conversion group in one shard and fails with
    # "could not revert some weight conversions because of offloading".
    # We want the compressed-tensors layout on disk anyway, not the original one.
    # Probe the *unwrapped* transformers signature: model.save_pretrained is
    # llmcompressor's wrapper, which only exposes **kwargs.
    save_kwargs = dict(save_compressed=True)
    base_params = inspect.signature(transformers.PreTrainedModel.save_pretrained).parameters
    if "save_original_format" in base_params:
        save_kwargs["save_original_format"] = False
    logger.info("save_pretrained kwargs: %s", save_kwargs)
    model.save_pretrained(output_dir, **save_kwargs)

    logger.info("Copying base-model auxiliary files from %s", model_path)
    copied = []

    def _ignore(src, files):
        skip = []
        for f in files:
            if f == "config.json" or "safetensors" in f or f in {".cache", "crc32.txt"}:
                skip.append(f)
            else:
                copied.append(f)
        return skip

    shutil.copytree(model_path, output_dir, ignore=_ignore, dirs_exist_ok=True)
    logger.info("Copied %d auxiliary files: %s", len(copied), sorted(set(copied)))
    postprocess_config(output_dir / "config.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PTQ for Qwen3.8-27B via llmcompressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Paths")
    g.add_argument("--model_path", required=True)
    g.add_argument("--output_path", required=True)

    g = p.add_argument_group("Scheme")
    g.add_argument("--strategy", default="fp8", choices=list(_STRATEGIES),
                   help="Quantization strategy")
    g.add_argument("--scheme", default="",
                   help="Override the llmcompressor scheme name implied by --strategy")
    g.add_argument("--ignore", default="",
                   help="Comma-separated extra module names/regexes to hold at bf16")
    g.add_argument("--allow_unsupported_gpu", action="store_true",
                   help="Proceed even if the GPU is below the scheme's compute capability")
    g.add_argument("--smoothquant", action=argparse.BooleanOptionalAction, default=True,
                   help="Run SmoothQuant before quantizing, as the Cosmos-Reason2 "
                        "pipeline does. Requires calibration data even for otherwise "
                        "data-free schemes")
    g.add_argument("--smoothing-strength", type=float, default=0.8,
                   help="SmoothQuant alpha (Cosmos-Reason2 uses 0.8)")

    g = p.add_argument_group("Calibration")
    g.add_argument("--dataset_id", default="abisee/cnn_dailymail")
    g.add_argument("--dataset_config", default="3.0.0")
    g.add_argument("--dataset_split", default="train")
    g.add_argument("--text_field", default="article,highlights,text")
    g.add_argument("--num_calib_samples", type=int, default=512)
    g.add_argument("--max_seq_len", type=int, default=2048)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Hardware")
    g.add_argument("--device", default="auto", help="device_map strategy")
    g.add_argument("--max_memory_per_gpu", type=int, default=0,
                   help="Per-GPU VRAM cap in GiB (0 = let accelerate decide)")
    g.add_argument("--cpu_offload_gb", type=int, default=0,
                   help="CPU RAM in GiB to offer accelerate as overflow")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme, needs_calib, min_cc, note = _STRATEGIES[args.strategy]
    if args.scheme:
        scheme = args.scheme
        logger.warning("Scheme overridden to %s", scheme)

    cc = detect_compute_capability()
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or ["CPU"]
    logger.info("GPUs: %s | min compute capability %.1f", ", ".join(names), cc)
    logger.info("Strategy %s -> scheme %s", args.strategy, scheme)
    logger.info("  %s", note)

    if cc < min_cc:
        msg = (f"{args.strategy} targets compute capability >= {min_cc:.1f}; this GPU "
               f"reports {cc:.1f}. The checkpoint will still be produced and is "
               f"numerically valid, but it will not run at full speed here.")
        if args.allow_unsupported_gpu:
            logger.warning(msg)
        else:
            logger.error(msg + "  Pass --allow_unsupported_gpu to continue anyway.")
            sys.exit(2)

    max_memory = None
    if args.max_memory_per_gpu > 0:
        max_memory = {i: f"{args.max_memory_per_gpu}GiB" for i in range(torch.cuda.device_count())}
        if args.cpu_offload_gb > 0:
            max_memory["cpu"] = f"{args.cpu_offload_gb}GiB"

    # NVFP4 carries a per-tensor global scale that must be shared across every
    # module vLLM packs together. Only the FP4 schemes have one.
    if "NVFP4" in scheme.upper():
        patch_fused_layer_names()

    model, config = load_model(args.model_path, args.device, max_memory)

    extra = [s.strip() for s in args.ignore.split(",") if s.strip()]
    ignore_list = build_ignore_list(model, extra)
    sequential_targets = detect_sequential_targets(model, config)

    # Recipe follows the Cosmos-Reason2 shape: SmoothQuant first to migrate
    # activation outliers into the weights, then the quantizer. Cosmos runs this
    # as pass 1 (LLM) and a bare QuantizationModifier as pass 2 (ViT); here the
    # whole vision tower stays bf16 — the reference config excludes every
    # `visual.*` module — so pass 2 is a no-op and only pass 1 runs.
    recipe = []
    if args.smoothquant:
        logger.info("SmoothQuant enabled (strength=%.2f)", args.smoothing_strength)
        sq_mappings = build_smoothquant_mappings(model)
        recipe.append(SmoothQuantModifier(
            smoothing_strength=args.smoothing_strength,
            mappings=sq_mappings,
        ))
    else:
        logger.warning("SmoothQuant disabled — this departs from the Cosmos recipe")
    recipe.append(QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore_list))

    oneshot_kwargs = dict(model=model, recipe=recipe)
    if sequential_targets:
        oneshot_kwargs["sequential_targets"] = sequential_targets

    # SmoothQuant fits its migration scales from activations, so it needs the
    # calibration set even when the quantization scheme itself is data-free.
    if args.smoothquant and not needs_calib:
        logger.info("Scheme %s is data-free, but SmoothQuant needs activations — "
                    "calibrating anyway.", scheme)
        needs_calib = True

    if needs_calib:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        calib = build_calibration_dataset(
            tokenizer, args.dataset_id, args.dataset_config or None, args.dataset_split,
            args.text_field, args.num_calib_samples, args.max_seq_len, args.seed,
        )
        oneshot_kwargs.update(
            dataset=calib,
            data_collator=data_collator,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.num_calib_samples,
        )
    else:
        logger.info("Scheme %s is data-free — skipping calibration.", scheme)

    logger.info("Running oneshot ...")
    oneshot(**oneshot_kwargs)

    save_checkpoint(model, model_path, output_dir)

    manifest = {
        "base_model": str(model_path),
        "strategy": args.strategy,
        "scheme": scheme,
        "smoothquant": {
            "enabled": args.smoothquant,
            "smoothing_strength": args.smoothing_strength,
            "num_mappings": len(sq_mappings),
        } if args.smoothquant else {"enabled": False},
        "calibration": None if not needs_calib else {
            "dataset_id": args.dataset_id,
            "dataset_config": args.dataset_config,
            "split": args.dataset_split,
            "num_samples": args.num_calib_samples,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
        "sequential_targets": sequential_targets,
        "num_ignored_modules": len(ignore_list),
        "ignore_patterns": summarize_ignore(ignore_list),
    }
    with open(output_dir / "quantization_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done. Quantized checkpoint at %s", output_dir)


if __name__ == "__main__":
    main()
