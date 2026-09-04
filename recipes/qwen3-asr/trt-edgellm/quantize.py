# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
Quantize the LLM component (thinker.model) of Qwen3-ASR-1.7B using NVIDIA ModelOpt.

Supported formats:
  int8  — INT8 SmoothQuant  (mtq.INT8_SMOOTHQUANT_CFG)
  int4  — INT4 AWQ           (mtq.INT4_AWQ_CFG)

audio_tower and lm_head are always excluded from quantization.

Requirements:
  datasets==2.19.0
  modelopt torch soundfile

Usage example:
  python quantize.py \\
      --model_path    /path/to/Qwen3-ASR-1.7B \\
      --qwen_asr_root /path/to/Qwen3-ASR \\
      --format        int8 \\
      --output_dir    ./Qwen3-ASR-1.7B-int8
"""

import sys
import io
import argparse
import json
import os
import shutil
import torch
import datasets
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from contextlib import redirect_stdout
from datasets import load_dataset

datasets.config.AUDIO_DECODER = "soundfile"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True,
                   help="Path to the original Qwen3-ASR-1.7B checkpoint")
    p.add_argument("--qwen_asr_root", required=True,
                   help="Path to the Qwen3-ASR repo root")
    p.add_argument("--format",        required=True, choices=["int8", "int4"],
                   help="int8 = INT8 SmoothQuant  |  int4 = INT4 AWQ")
    p.add_argument("--output_dir",    required=True,
                   help="Directory to save the quantized HF checkpoint")
    p.add_argument("--device",        default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _take_fleurs(lang: str, n: int) -> list:
    try:
        ds = load_dataset("google/fleurs", lang, split="validation", streaming=True)
        return [
            {
                "array": s["audio"]["array"],
                "sampling_rate": s["audio"]["sampling_rate"],
                "text": s["transcription"],
            }
            for s in list(ds.take(n))
        ]
    except Exception as e:
        print(f"[WARNING] Skipping {lang}: {e}")
        return []


def _take_librispeech(split: str, n: int) -> list:
    try:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
        return [
            {
                "array": s["audio"]["array"],
                "sampling_rate": s["audio"]["sampling_rate"],
                "text": s["text"],
            }
            for s in list(ds.take(n))
        ]
    except Exception as e:
        print(f"[WARNING] Skipping librispeech {split}: {e}")
        return []


_MULTI_LANGS = [
    "yue_hant_hk", "ar_eg", "de_de", "fr_fr", "es_419",
    "pt_br", "id_id", "it_it", "ko_kr", "ru_ru",
    "th_th", "vi_vn", "ja_jp", "tr_tr", "hi_in",
    "ms_my", "nl_nl", "sv_se", "da_dk", "fi_fi",
    "pl_pl", "cs_cz", "fil_ph", "fa_ir", "el_gr",
    "ro_ro", "hu_hu", "mk_mk",
]
_TARGET_TOTAL = 512


def build_calibration_data() -> list:
    """
    Mirror the Qwen3-ASR training distribution:
      35% EN   : ~179 samples — LibriSpeech train.100
      35% ZH   : ~179 samples — FLEURS cmn_hans_cn
      30% Multi: ~154 samples — 28 languages (dynamic distribution)
    Total: 512 samples
    """
    en_total    = int(_TARGET_TOTAL * 0.35)
    zh_total    = int(_TARGET_TOTAL * 0.35)
    multi_total = _TARGET_TOTAL - en_total - zh_total

    en_samples    = _take_librispeech("train.100", en_total)
    zh_samples    = _take_fleurs("cmn_hans_cn", zh_total)
    multi_samples: list = []
    base_per_lang = multi_total // len(_MULTI_LANGS)
    remainder     = multi_total % len(_MULTI_LANGS)
    for i, lang in enumerate(_MULTI_LANGS):
        n = base_per_lang + (1 if i < remainder else 0)
        multi_samples += _take_fleurs(lang, n)
    return en_samples + zh_samples + multi_samples


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def build_quant_cfg(fmt: str) -> dict:
    if fmt == "int8":
        cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
    else:
        cfg = mtq.INT4_AWQ_CFG.copy()
    cfg["quant_cfg"]["*audio_tower*"] = {"enable": False}
    cfg["quant_cfg"]["*lm_head*"]     = {"enable": False}
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


def copy_base_metadata(model_path: str, output_dir: str) -> None:
    """Copy every non-weight file from the base checkpoint verbatim.

    This deliberately replaces `processor.save_pretrained()`. That call
    re-serializes from the live Python object, so whatever the loaded class does
    not model is silently dropped, and whatever load-time flag was passed is
    baked in — this processor is constructed with `fix_mistral_regex=True`, and
    saving it writes that mutated tokenizer back out as if it were the base.

    The feature extractor config is the part that matters at inference: it
    carries the mel/window parameters the audio tower was trained against. If it
    is rewritten or dropped, the checkpoint silently preprocesses audio
    differently from the model it was derived from.

    Copying the originals is lossless and does not depend on the transformers
    version in the environment. `config.json` is excluded so that
    `patch_config_json` still operates on the config the quantized model wrote.
    """
    copied = []

    def _ignore(src, names):
        skip = []
        for name in names:
            (skip if _skip_from_base(name) else copied).append(name)
        return skip

    shutil.copytree(model_path, output_dir, ignore=_ignore, dirs_exist_ok=True)
    print(f"[COPY] {len(copied)} base-model metadata files from {model_path}: "
          f"{sorted(set(copied))}")


# ---------------------------------------------------------------------------
# config.json patch
# ---------------------------------------------------------------------------

def patch_config_json(output_dir: str) -> None:
    """
    Fix thinker_config.model_type after ModelOpt save_pretrained.
    ModelOpt overwrites model_type with an internal value; restore it
    to "qwen3_asr" so that TensorRT-Edge-LLM can load the checkpoint.
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    current = cfg.get("thinker_config", {}).get("model_type", "")
    if current == "qwen3_asr":
        print("[PATCH] config.json thinker_config.model_type already correct — skipped.")
        return

    cfg["thinker_config"]["model_type"] = "qwen3_asr"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[PATCH] config.json thinker_config.model_type: '{current}' → 'qwen3_asr'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sys.path.insert(0, args.qwen_asr_root)

    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )

    print(f"[CONFIG] format={args.format}  output={args.output_dir}")

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.float16, device_map=args.device
    )
    processor = Qwen3ASRProcessor.from_pretrained(
        args.model_path, fix_mistral_regex=True
    )
    device = next(model.parameters()).device

    # Untie lm_head from embed_tokens before quantization
    model.thinker.lm_head.weight = torch.nn.Parameter(
        model.thinker.lm_head.weight.data.clone()
    )

    calib_data = build_calibration_data()
    print(f"Total calibration samples: {len(calib_data)}")

    def forward_step(model_to_run, data):
        inputs = processor(
            audio=data["array"],
            sampling_rate=data["sampling_rate"],
            text=data["text"],
            return_tensors="pt",
        )
        inputs = {
            k: v.to(dtype=torch.float16, device=device) if v.dtype == torch.float32
            else v.to(device=device)
            for k, v in inputs.items()
        }
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return model_to_run.thinker(
            input_ids=inputs["input_ids"],
            input_features=inputs.get("input_features"),
            attention_mask=inputs.get("attention_mask"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            labels=labels,
        )

    cfg = build_quant_cfg(args.format)

    def forward_loop(model_to_quantize):
        for data in calib_data:
            forward_step(model_to_quantize, data)

    # Enable HF-compatible checkpointing before quantization
    mto.enable_huggingface_checkpointing()

    model = mtq.quantize(model, cfg, forward_loop)

    # Log quantization decisions
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"quantize_decisions_{args.format}.txt")
    buf = io.StringIO()
    with redirect_stdout(buf):
        mtq.print_quant_summary(model)
    with open(log_path, "w") as f:
        f.write(buf.getvalue())
    print(f"[LOG] Quantization decisions: {log_path}")

    # Save checkpoint
    if hasattr(model, "generation_config"):
        if not model.generation_config.do_sample:
            model.generation_config.temperature = None
            model.generation_config.top_p = None

    model.save_pretrained(args.output_dir)
    copy_base_metadata(args.model_path, args.output_dir)
    print(f"[SAVED] Checkpoint: {args.output_dir}")

    # Fix thinker_config.model_type overwritten by ModelOpt
    patch_config_json(args.output_dir)
    print(f"[DONE] Quantized checkpoint ready: {args.output_dir}")


if __name__ == "__main__":
    main()
