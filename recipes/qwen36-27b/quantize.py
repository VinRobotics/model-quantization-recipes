# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause

"""
quantize.py — Multi-scheme quantization for Qwen3.6-27B using llm-compressor.

Supported schemes: fp8, nvfp4, int8, int4
Scheme configurations are loaded from configs/schemes.yaml (no hardcoding).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCHEMES_CONFIG = Path(__file__).parent / "configs" / "schemes.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.6-27B using llm-compressor (multi-scheme)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path (local) or Hub ID of the source model",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Destination directory for the quantized model. "
            "Defaults to <model_path>-<SCHEME> (e.g. Qwen3.6-27B-FP8)"
        ),
    )

    # Scheme
    parser.add_argument(
        "--scheme",
        default="fp8",
        help=(
            "Quantization scheme key defined in configs/schemes.yaml. "
            "Built-in options: fp8 | nvfp4 | int8 | int4"
        ),
    )
    parser.add_argument(
        "--schemes_config",
        default=str(DEFAULT_SCHEMES_CONFIG),
        help="Path to the YAML file containing scheme definitions",
    )

    # Calibration dataset
    parser.add_argument(
        "--dataset_id",
        default="abisee/cnn_dailymail",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--dataset_config",
        default="3.0.0",
        help="Dataset config / version string",
    )
    parser.add_argument(
        "--dataset_split",
        default="train",
        help="Dataset split to use for calibration",
    )
    parser.add_argument(
        "--dataset_text_field",
        default="article",
        help="Field name containing the text in the dataset",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max token length per calibration sample",
    )

    # Hardware
    parser.add_argument(
        "--device",
        default="auto",
        help="device_map strategy passed to from_pretrained",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=int,
        default=28,
        help="Max VRAM cap per GPU in GiB (RTX 5090 = 32 GiB; leave headroom)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scheme config loader
# ---------------------------------------------------------------------------

def load_scheme_config(schemes_config_path: str, scheme_key: str) -> dict:
    """Load a single scheme definition from the YAML config file."""
    config_path = Path(schemes_config_path)
    if not config_path.exists():
        print(f"[ERROR] Scheme config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open() as fh:
        all_schemes = yaml.safe_load(fh)

    if scheme_key not in all_schemes:
        available = ", ".join(all_schemes.keys())
        print(
            f"[ERROR] Unknown scheme '{scheme_key}'. "
            f"Available in {config_path}: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    return all_schemes[scheme_key]


def build_recipe(scheme_cfg: dict) -> QuantizationModifier:
    """Construct a QuantizationModifier from a scheme config dict."""
    return QuantizationModifier(
        targets=scheme_cfg.get("targets", "Linear"),
        scheme=scheme_cfg["scheme"],
        ignore=scheme_cfg.get("ignore", []),
    )


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
    version in the environment.
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
# Calibration dataset helpers
# ---------------------------------------------------------------------------

def build_calib_dataset(
    processor,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    text_field: str,
    num_samples: int,
    max_seq_len: int,
) -> Dataset:
    """Stream, tokenise and return a calibration dataset."""
    print(f"[INFO] Loading dataset '{dataset_id}' (streaming)...")
    ds = load_dataset(
        dataset_id,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=num_samples * 3).take(num_samples)

    samples = []
    print(f"[INFO] Tokenising {num_samples} samples: ", end="", flush=True)
    for i, item in enumerate(ds):
        if (i + 1) % 50 == 0:
            print(f"{i + 1}", end=" ", flush=True)

        text = item[text_field].strip()
        enc = processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        samples.append(
            {
                "input_ids": enc["input_ids"][0].tolist(),
                "attention_mask": enc["attention_mask"][0].tolist(),
            }
        )

    print(f"\n[INFO] Loaded {len(samples)} calibration samples.")
    return Dataset.from_list(samples)


def data_collator(batch):
    assert len(batch) == 1, "Batch size must be 1 for calibration."
    return {k: torch.tensor(v).unsqueeze(0) for k, v in batch[0].items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Resolve output path ──────────────────────────────────────────────
    if args.output_path is None:
        model_basename = Path(args.model_path).name
        args.output_path = str(
            Path(args.model_path).parent / f"{model_basename}-{args.scheme.upper()}"
        )
        print(f"[INFO] output_path not specified — using: {args.output_path}")

    # ── Load scheme config ───────────────────────────────────────────────
    scheme_cfg = load_scheme_config(args.schemes_config, args.scheme)
    print(
        f"[INFO] Scheme: {args.scheme.upper()} "
        f"(llm-compressor scheme string: '{scheme_cfg['scheme']}')"
    )

    # ── GPU inventory ────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{args.max_memory_per_gpu}GiB" for i in range(num_gpus)}
    print(f"[INFO] Detected {num_gpus} GPU(s); VRAM cap: {args.max_memory_per_gpu} GiB each")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"[INFO] Loading model from: {args.model_path}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device,
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # ── Build recipe ─────────────────────────────────────────────────────
    recipe = build_recipe(scheme_cfg)

    # ── Calibration dataset ──────────────────────────────────────────────
    calib_dataset = build_calib_dataset(
        processor=processor,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_field=args.dataset_text_field,
        num_samples=args.num_calib_samples,
        max_seq_len=args.max_seq_len,
    )

    # ── Quantize ─────────────────────────────────────────────────────────
    print(f"[INFO] Starting {args.scheme.upper()} quantisation…")
    os.makedirs(args.output_path, exist_ok=True)

    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_dataset,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calib_samples,
        data_collator=data_collator,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    # Weights and config.json come from the quantized model; every other
    # metadata file is copied from the base checkpoint rather than
    # re-serialized. See copy_base_metadata for why.
    model.save_pretrained(args.output_path)
    copy_base_metadata(args.model_path, args.output_path)
    print(f"[INFO] Done. Quantised model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
