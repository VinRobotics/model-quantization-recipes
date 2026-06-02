#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
quantize.py — INT8 / FP8 / NVFP4 quantization via nvidia-modelopt.
"""

import argparse
import io
import os
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
    tokenizer.save_pretrained(args.output_path)

    print(f"\nDone. Output: {args.output_path}")


if __name__ == "__main__":
    main()
