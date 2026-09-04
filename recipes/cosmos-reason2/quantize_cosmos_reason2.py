# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import shutil

import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",         required=True)
    parser.add_argument("--output_path",        required=True)
    parser.add_argument("--num_calib_samples",  type=int, default=512)
    parser.add_argument("--max_seq_len",        type=int, default=1024)
    parser.add_argument("--dataset_id",         default="abisee/cnn_dailymail")
    parser.add_argument("--dataset_config",     default="3.0.0")
    parser.add_argument("--dataset_split",      default="train")
    parser.add_argument("--device",             default="auto")
    parser.add_argument("--max_memory_per_gpu", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{args.max_memory_per_gpu}GiB" for i in range(num_gpus)}
    print(f"GPUs: {num_gpus}, VRAM cap: {args.max_memory_per_gpu} GiB each")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device,
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Quantize LLM backbone only (196 Linear layers, 28 layers x 7)
    # Visual encoder (blocks 0-23 + merger + deepstack) kept in bf16
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*visual.*",
            "re:.*lm_head",
        ],
    )

    # Text-only calibration: the vision tower is ignored entirely, so multimodal
    # inputs would add nothing to the activation statistics.
    print(f"Loading dataset '{args.dataset_id}' (streaming)...")
    ds = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=args.num_calib_samples * 3).take(args.num_calib_samples)

    samples = []
    print(f"Tokenizing {args.num_calib_samples} samples: ", end="", flush=True)
    for i, item in enumerate(ds):
        if (i + 1) % 50 == 0:
            print(f"{i + 1} ", end="", flush=True)
        text = item["article"].strip()
        inputs = processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        samples.append({
            "input_ids":      inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
        })
    print(f"\nLoaded {len(samples)} samples.")

    calib_dataset = Dataset.from_list(samples)

    def data_collator(batch):
        assert len(batch) == 1
        return {key: torch.tensor(value).unsqueeze(0) for key, value in batch[0].items()}

    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_dataset,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calib_samples,
        data_collator=data_collator,
    )

    # Weights and config.json come from the quantized model; every other metadata
    # file is copied from the base checkpoint rather than re-serialized.
    model.save_pretrained(args.output_path)
    copy_base_metadata(args.model_path, args.output_path)
    print(f"Done. Saved to {args.output_path}")


if __name__ == "__main__":
    main()
