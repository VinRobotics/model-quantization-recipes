#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Quantize Qwen2.5-VL with NVIDIA ModelOpt.

Usage:
    python quantize.py --strategy s1 --cfg fp8_default
    python quantize.py --strategy s3 --cfg nvfp4_awq_full --dry-run

Supports 4 strategies x 5 schemes; see ``configs/schemes.yaml`` for the matrix
and which combinations are blocked or experimental on this hardware.
"""

import argparse
import os
import sys
import time

import modelopt.torch.quantization as mtq
import torch

from calibration import (
    multimodal_calib_dataloader,
    text_calib_dataloader,
    vln_calib_dataloader,
)
from quant_schemes import (
    build_quant_config,
    calib_batch_size,
    describe,
    load_registry,
    scheme_names,
    strategy_names,
    validate,
)
from model_loader import (
    calibrate_multimodal,
    calibrate_text,
    export_quantized_model,
    load_model,
)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_DIR = os.path.expanduser(
    os.environ.get("INTERNVLA_CKPT", "~/InternNav/checkpoints/InternVLA-N1-DualVLN")
)
DEFAULT_OUTPUT_BASE = os.path.expanduser(
    os.environ.get("VLN_OPT_WORK", "~/vln-opt-work")
)
TEXT_DATASET = "abisee/cnn_dailymail"
MULTIMODAL_DATASET = "lmms-lab/MMMU"
MULTIMODAL_MAX_SAMPLES = 128  # NVIDIA's cap — VLM calibration is GPU-mem bound
# In-distribution calibration: the InternData-N1 VLN-CE subset System 2 was
# fine-tuned on. Override with --calib-data or $VLN_CALIB_DATA.
VLN_CALIB_DATA = os.path.expanduser(
    os.environ.get(
        # Default = the output of build/00_fetch_calib_scenes.sh. Point it at any
        # InternData-N1 VLN-CE tree (a scene dir or a parent of several) via the env var.
        "VLN_CALIB_DATA",
        "~/vln-opt-work/calib_scenes",
    )
)
VLN_RGB_KEY = os.environ.get("VLN_RGB_KEY", "observation.images.rgb.125cm_0deg")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2.5-VL with NVIDIA ModelOpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=describe(),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=strategy_names(),
        help="Quantization strategy (see options below).",
    )
    parser.add_argument(
        "--scheme",
        required=True,
        choices=scheme_names(),
        help="Quantization scheme (see options below).",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_DIR,
        help=f"Path to source HF model (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output dir for quantized model. Default: "
        f"{DEFAULT_OUTPUT_BASE}/qwen2.5-vl-7b-<strategy>-<cfg>/",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=512,
        help="Calibration sample count. Capped at 128 for multimodal "
        "strategies (S3/S4) per NVIDIA's reference. Default: 512.",
    )
    parser.add_argument(
        "--calib",
        default="auto",
        choices=["auto", "text", "multimodal", "vln"],
        help="Calibration data source. 'auto' (default) keeps the legacy "
        "behaviour: text (cnn_dailymail) for LLM-only strategies, multimodal "
        "(MMMU) when the visual tower is quantized. 'vln' uses the "
        "in-distribution InternData-N1 VLN-CE set (recommended for this model).",
    )
    parser.add_argument(
        "--calib_data_root",
        default=VLN_CALIB_DATA,
        help=f"Root of the VLN-CE LeRobot data for --calib vln "
        f"(default: {VLN_CALIB_DATA}).",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp16", "bf16"],
        help="Model load dtype. Default: bf16 (matches source config).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model + calibration. Default: cuda.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="DIR",
        help="Enable layerwise calibration resume from DIR (useful for "
        "AWQ/Hessian recovery on crash).",
    )
    parser.add_argument(
        "--allow_experimental",
        action="store_true",
        help="Permit schemes marked experimental in configs/schemes.yaml. NVFP4 needs "
             "this: it quantizes and generates fluent text, but the System 2 -> System 1 "
             "bridge breaks (z_latents cosine 0.647 vs 0.9956 for FP8).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load model + validate environment, skip quantization. "
        "Useful for verifying setup before a long run.",
    )
    return parser.parse_args()


def derive_output_path(args: argparse.Namespace) -> str:
    if args.output_path is not None:
        return args.output_path
    tag = f"internvla-n1-system2-{args.strategy}-{args.scheme}"
    return os.path.join(DEFAULT_OUTPUT_BASE, tag)


def print_environment() -> None:
    """Quick environment summary for logging at the top of every run."""
    print("=" * 70)
    print("Environment:")
    print(f"  torch:    {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"  device:   {props.name}")
        print(f"  arch:     sm_{cc[0]}{cc[1]}")
        print(f"  VRAM:     {props.total_memory / 1e9:.1f} GB")
        print(f"  arch list: {torch.cuda.get_arch_list()}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    output_path = derive_output_path(args)

    print_environment()

    registry = load_registry()

    # Reject impossible combinations before the model is loaded: a bad request should
    # cost seconds, not a checkpoint load followed by a mid-quantization crash.
    try:
        validate(args.scheme, args.strategy, model_path=args.model_path,
                 allow_experimental=args.allow_experimental, registry=registry)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    strat = registry["strategies"][args.strategy]
    cfg_info = registry["schemes"][args.scheme]

    # Resolve calibration source. 'auto' preserves the legacy behaviour; an
    # explicit --calib overrides it (e.g. VLN calib for an LLM-only strategy).
    calib_mode = args.calib
    if calib_mode == "auto":
        calib_mode = "multimodal" if strat["quantize_visual"] else "text"
    calib_desc = {
        "text": "text-only (cnn_dailymail)",
        "multimodal": "multimodal (MMMU image+text)",
        "vln": "VLN in-distribution (InternData-N1 VLN-CE)",
    }[calib_mode]

    print()
    print(f"Strategy:       {args.strategy} — {strat['description']}")
    print(f"CFG preset:     {args.scheme} — {cfg_info['description']}")
    print(f"Model path:     {args.model_path}")
    print(f"Output path:    {output_path}")
    print(f"Calibration:    {calib_desc}")
    print()

    if not os.path.isdir(args.model_path):
        print(f"ERROR: model_path not found: {args.model_path}", file=sys.stderr)
        return 1

    # -- Load model ------------------------------------------------------- #
    t_load_start = time.time()
    print("[1/4] Loading model...")
    model, tokenizer, processor = load_model(
        args.model_path, dtype=args.dtype, device=args.device
    )
    original_max_length = tokenizer.model_max_length
    print(f"      done in {time.time() - t_load_start:.1f}s")
    print(f"      model class: {type(model).__name__}")
    print(f"      tokenizer.model_max_length (preserved): {original_max_length}")

    if args.dry_run:
        print("[dry-run] Skipping quantization. Environment OK.")
        return 0

    # -- Build quant config ---------------------------------------------- #
    quant_cfg = build_quant_config(
        scheme=args.scheme,
        strategy=args.strategy,
        layerwise_checkpoint_dir=args.resume,
    )

    # -- Build dataloader & forward loop --------------------------------- #
    t_calib_start = time.time()
    print("[2/4] Preparing calibration data...")

    if calib_mode in ("multimodal", "vln"):
        if processor is None:
            raise RuntimeError(
                f"{calib_mode} calibration requires an AutoProcessor but none "
                "was found in the model dir."
            )
        # Both image paths are GPU-memory bound (multi-image forward), so they
        # share NVIDIA's multimodal sample cap.
        mm_samples = min(args.num_calib_samples, MULTIMODAL_MAX_SAMPLES)
        if mm_samples < args.num_calib_samples:
            print(f"      capping num_samples {args.num_calib_samples} → "
                  f"{mm_samples} (multimodal GPU-mem cap)")
        if calib_mode == "vln":
            batches = vln_calib_dataloader(
                processor,
                data_root=args.calib_data_root,
                num_samples=mm_samples,
                rgb_key=VLN_RGB_KEY,
            )
            print(f"      VLN batches prepared: {len(batches)} "
                  f"(data: {args.calib_data_root})")
        else:
            batches = multimodal_calib_dataloader(
                processor,
                dataset_name=MULTIMODAL_DATASET,
                num_samples=mm_samples,
            )
            print(f"      multimodal batches prepared: {len(batches)}")
        forward_loop = lambda m: calibrate_multimodal(m, batches)  # noqa: E731
    else:
        batch_size = calib_batch_size(args.scheme, is_image_calib=False)
        loader = text_calib_dataloader(
            tokenizer,
            dataset_name=TEXT_DATASET,
            batch_size=batch_size,
            num_samples=args.num_calib_samples,
        )
        forward_loop = lambda m: calibrate_text(m, loader)  # noqa: E731
        print(f"      text loader prepared: batch_size={batch_size}, "
              f"samples={args.num_calib_samples}")

    # -- Quantize -------------------------------------------------------- #
    print("[3/4] Running quantization...")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)
    print(f"      quantization done in {time.time() - t_calib_start:.1f}s")

    # Safety: tokenizer.model_max_length must not have been touched by
    # calibration. This is the user's main concern about preprocessor leakage.
    assert tokenizer.model_max_length == original_max_length, (
        f"tokenizer.model_max_length changed during calibration: "
        f"{original_max_length} → {tokenizer.model_max_length}"
    )

    # -- Export ---------------------------------------------------------- #
    t_export_start = time.time()
    print("[4/4] Exporting quantized checkpoint...")
    export_quantized_model(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        model_dir=args.model_path,
        output_dir=output_path,
    )
    print(f"      export done in {time.time() - t_export_start:.1f}s")
    print()
    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())