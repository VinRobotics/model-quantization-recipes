#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
quantize.py — INT8 / FP8 / NVFP4 quantization via nvidia-modelopt.
"""

import argparse
import io
import os
import shutil
from contextlib import redirect_stdout

import torch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration
from datasets import load_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize model via nvidia-modelopt")
    parser.add_argument("--model_path",          required=True)
    parser.add_argument("--output_path",         required=True)
    parser.add_argument("--num_calib_samples",   type=int, default=256)
    parser.add_argument("--max_seq_len",         type=int, default=1024)
    parser.add_argument("--device",              default="auto")
    parser.add_argument("--quant_dtype",         default="nvfp4",
                        choices=["int8", "fp8", "nvfp4"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calib(tokenizer, n: int, max_len: int) -> list:
    ds = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=n * 3).take(n)

    samples = []
    for item in ds:
        inputs = tokenizer(
            item["article"].strip(),
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        samples.append(inputs["input_ids"])

    return samples


def make_forward_loop(calib: list):
    def forward_loop(model):
        for ids in calib:
            with torch.no_grad():
                model(input_ids=ids.to(model.device))
            torch.cuda.empty_cache()
    return forward_loop


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def build_quant_config(model, dtype: str = "nvfp4") -> dict:
    if dtype == "int8":
        cfg = mtq.INT8_DEFAULT_CFG.copy()
    elif dtype == "fp8":
        cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif dtype == "nvfp4":
        cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    exclude_patterns = [
        "lm_head",
        "model.embed_tokens",
        "model.visual*",
    ]

    for name, _ in model.named_modules():
        if "norm" in name:
            exclude_patterns.append(name)
        if any(x in name for x in ["gate", "router", "shared_expert_gate"]):
            exclude_patterns.append(name)
        if "linear_attn" in name and any(
            x in name for x in ["conv1d", "A_log", "dt_bias", "in_proj"]
        ):
            exclude_patterns.append(name)

    for p in exclude_patterns:
        cfg["quant_cfg"][p] = {"enable": False}

    return cfg


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

# Files the quantized checkpoint owns and must not inherit from the base model:
# its own config.json, and its own weights.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack")
_SKIP_EXACT = {"config.json", ".cache", "crc32.txt"}


def _skip_from_base(name: str) -> bool:
    return (
        name in _SKIP_EXACT
        or name.endswith(_WEIGHT_SUFFIXES)
        or ".safetensors.index" in name
        or ".bin.index" in name
    )


def copy_base_metadata(model_path: str, output_path: str) -> None:
    """Copy every non-weight file from the base checkpoint verbatim.

    This deliberately replaces `processor.save_pretrained()` /
    `tokenizer.save_pretrained()`. Those re-serialize from the live Python
    object, so whatever the loaded class does not model is silently dropped.
    Measured on a Qwen3-family VLM processor: the saved directory contains no
    `preprocessor_config.json` and no `video_preprocessor_config.json` at all,
    and its `tokenizer_config.json` comes back without `added_tokens_decoder`
    or `additional_special_tokens`.

    The consequence is not a load error. The image processor falls back to
    library defaults whose pixel budget differs from the base model's, so the
    quantized checkpoint preprocesses inputs differently from the model it was
    derived from — and that only shows up at inference, on inputs larger than
    the ones the calibration pass happened to exercise.

    Copying the originals is lossless and does not depend on the transformers
    version in the environment. `export_hf_checkpoint` has already written
    `config.json`, `hf_quant_config.json` and the weight shards; neither is
    overwritten here.
    """
    copied = []

    def _ignore(src, names):
        skip = []
        for name in names:
            (skip if _skip_from_base(name) else copied).append(name)
        return skip

    shutil.copytree(model_path, output_path, ignore=_ignore, dirs_exist_ok=True)
    print(f"[INFO] Copied {len(copied)} base-model metadata files from {model_path}: "
          f"{sorted(set(copied))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("[1/5] Loading model...")
    mto.enable_huggingface_checkpointing()

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    model.eval()

    print("[2/5] Building quantization config...")
    quant_cfg = build_quant_config(model, dtype=args.quant_dtype)

    print(f"[3/5] Building calibration data ({args.num_calib_samples} samples)...")
    calib = build_calib(tokenizer, args.num_calib_samples, args.max_seq_len)

    print("[4/5] Quantizing...")
    model = mtq.quantize(model, quant_cfg, make_forward_loop(calib))

    os.makedirs(args.output_path, exist_ok=True)

    summary = io.StringIO()
    with redirect_stdout(summary):
        mtq.print_quant_summary(model)
    with open(os.path.join(args.output_path, "quant_log.txt"), "w") as f:
        f.write(summary.getvalue())

    print("[5/5] Exporting HF checkpoint...")
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=args.output_path)
    copy_base_metadata(args.model_path, args.output_path)

    print(f"\nDone. Output: {args.output_path}")


if __name__ == "__main__":
    main()
