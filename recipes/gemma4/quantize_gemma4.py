# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

DEFAULT_DATASET_ID = "abisee/cnn_dailymail"
DEFAULT_DATASET_CONFIG = "3.0.0"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_TEXT_COLUMN = "article"
DEFAULT_MAX_MEMORY_PER_GPU_GIB = 72
DEFAULT_MODEL_DTYPE = "bfloat16"
DEFAULT_QUANTIZATION = "fp8"
SUPPORTED_QUANTIZATION_CONFIGS: dict[str, Any] = {
    "fp8": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Gemma4 checkpoints with ModelOpt."
    )
    parser.add_argument("--model_path", required=True, help="Path or Hugging Face ID")
    parser.add_argument("--output_path", required=True, help="Directory for quantized output")
    parser.add_argument(
        "--quantization",
        choices=sorted(SUPPORTED_QUANTIZATION_CONFIGS),
        default=DEFAULT_QUANTIZATION,
        help="Quantization preset to apply.",
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float16", "bfloat16"],
        default=DEFAULT_MODEL_DTYPE,
        help="Base model dtype used while loading the source checkpoint.",
    )
    parser.add_argument("--num_calib_samples", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset_split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--dataset_text_column", default=DEFAULT_DATASET_TEXT_COLUMN)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max_memory_per_gpu",
        type=int,
        default=DEFAULT_MAX_MEMORY_PER_GPU_GIB,
        help="Max VRAM (GiB) per GPU reserved for model weights.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_mapping[dtype_name]


def is_vision_language_model(model_path: str) -> bool:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config_dict = config.to_dict()
    return "vision_config" in config_dict or "image_embd_layer" in config_dict.get(
        "embd_layer", {}
    )


def load_model_and_processor(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    max_memory: dict[int, str],
) -> tuple[Any, Any]:
    model_loader: type[Any]
    if is_vision_language_model(model_path):
        model_loader = AutoModelForImageTextToText
    else:
        model_loader = AutoModelForCausalLM

    model = model_loader.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


def get_text_tokenizer(processor: Any) -> Any:
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    return processor


def build_default_exclude_list(model_config: Any) -> list[str]:
    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        return ["lm_head"]

    num_layers = text_config.num_hidden_layers
    has_audio = getattr(model_config, "audio_config", None) is not None
    has_ple = getattr(text_config, "hidden_size_per_layer_input", 0) > 0

    exclude_patterns = ["lm_head", "model.vision_tower*", "model.embed_vision*"]

    if has_audio:
        exclude_patterns.extend(["model.audio_tower*", "model.embed_audio*"])

    if has_ple:
        exclude_patterns.extend(
            [
                "model.language_model.per_layer_model_projection",
                "model.language_model.per_layer_projection_norm",
                "*per_layer_input_gate*",
                "*per_layer_projection*",
                "*post_per_layer_input_norm*",
            ]
        )

    exclude_patterns.extend(
        f"model.language_model.layers.{layer_index}.self_attn*"
        for layer_index in range(num_layers)
    )
    return exclude_patterns


def build_calibration_samples(
    processor: Any,
    num_samples: int,
    max_seq_len: int,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    dataset_text_column: str,
) -> list[Any]:
    dataset_kwargs: dict[str, Any] = {
        "split": dataset_split,
        "streaming": True,
    }
    if dataset_config:
        dataset_kwargs["name"] = dataset_config

    # `datasets.load_dataset` is typed as returning several dataset container variants.
    # With `split=...` and `streaming=True` we expect an iterable dataset here, but
    # static checkers still keep the broader union, so narrow it locally.
    dataset_stream = cast(Any, load_dataset(dataset_id, **dataset_kwargs))
    shuffled_samples = cast(
        Any,
        dataset_stream.shuffle(seed=42, buffer_size=num_samples * 3).take(num_samples),
    )

    tokenizer = get_text_tokenizer(processor)
    calibration_samples: list[Any] = []
    for item in shuffled_samples:
        text = str(item[dataset_text_column]).strip()
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        calibration_samples.append(inputs["input_ids"])

    return calibration_samples


def build_quantization_config(quantization: str, exclude_list: list[str]) -> dict[str, Any]:
    quant_config = cast(dict[str, Any], SUPPORTED_QUANTIZATION_CONFIGS[quantization].copy())
    quant_cfg = cast(dict[str, Any], quant_config["quant_cfg"])
    for pattern in exclude_list:
        quant_cfg[pattern] = {"enable": False}
    return quant_config


def make_forward_loop(calibration_samples: list[Any]) -> Callable[[Any], None]:
    def forward_loop(model: Any) -> None:
        for input_ids in calibration_samples:
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                model(input_ids=input_ids)
            torch.cuda.empty_cache()

    return forward_loop


def write_quantization_summary(model: Any, output_dir: Path) -> Path:
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        mtq.print_quant_summary(model)

    quant_log_path = output_dir / "quantize_decisions.txt"
    quant_log_path.write_text(summary_buffer.getvalue(), encoding="utf-8")
    return quant_log_path


def main() -> None:
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    max_memory = {
        gpu_index: f"{args.max_memory_per_gpu}GiB" for gpu_index in range(num_gpus)
    }
    print(f"Detected {num_gpus} GPU(s). VRAM cap per GPU: {args.max_memory_per_gpu} GiB")

    model_path = str(args.model_path)
    output_path = Path(args.output_path)
    torch_dtype = get_torch_dtype(args.model_dtype)

    print(f"[1/4] Loading Gemma4 model from {model_path} ...")
    mto.enable_huggingface_checkpointing()
    model, processor = load_model_and_processor(
        model_path=model_path,
        dtype=torch_dtype,
        device=args.device,
        max_memory=max_memory,
    )
    model.eval()

    exclude_list = build_default_exclude_list(model.config)
    quantization_config = build_quantization_config(args.quantization, exclude_list)
    print(
        f"[2/4] Quantization preset: {args.quantization} with "
        f"{len(exclude_list)} default exclusions"
    )

    print(
        f"[3/4] Building calibration data ({args.num_calib_samples} samples, "
        f"max_seq={args.max_seq_len}) ..."
    )
    calibration_samples = build_calibration_samples(
        processor=processor,
        num_samples=args.num_calib_samples,
        max_seq_len=args.max_seq_len,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
    )
    print(f"Actual samples loaded: {len(calibration_samples)}")

    print("[4/4] Quantizing ...")
    model = mtq.quantize(
        model,
        quantization_config,
        make_forward_loop(calibration_samples),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    quant_log_path = write_quantization_summary(model, output_path)
    print(f"Quant summary saved to {quant_log_path}")

    print(f"Saving quantized model to {output_path} ...")
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(output_path))
    processor.save_pretrained(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
