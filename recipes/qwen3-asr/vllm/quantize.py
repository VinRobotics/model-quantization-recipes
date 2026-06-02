# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
import sys
import io
import argparse
import torch
import datasets
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from contextlib import redirect_stdout
from datasets import load_dataset

datasets.config.AUDIO_DECODER = "soundfile"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True,  help="Path to Qwen3-ASR-1.7B")
    p.add_argument("--qwen_asr_root", required=True,  help="Path to Qwen3-ASR repo root")
    p.add_argument("--output_pt",     required=True,  help="Output .pt checkpoint path")
    p.add_argument("--format",        required=True,  choices=["fp8", "nvfp4"],
                   help="Quantization format")
    p.add_argument("--algorithm",     default="max",  choices=["max", "mse", "awq_full"],
                   help="Calibration algorithm")
    p.add_argument("--device",        default="cuda:0")
    return p.parse_args()


def take_fleurs(lang: str, n: int) -> list:
    try:
        ds = load_dataset("google/fleurs", lang, split="validation", streaming=True)
        return [
            {"array": s["audio"]["array"],
             "sampling_rate": s["audio"]["sampling_rate"],
             "text": s["transcription"]}
            for s in list(ds.take(n))
        ]
    except Exception as e:
        print(f"[WARNING] Skipping {lang}: {e}")
        return []


def take_librispeech(split: str, n: int) -> list:
    try:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
        return [
            {"array": s["audio"]["array"],
             "sampling_rate": s["audio"]["sampling_rate"],
             "text": s["text"]}
            for s in list(ds.take(n))
        ]
    except Exception as e:
        print(f"[WARNING] Skipping librispeech {split}: {e}")
        return []


def build_quant_cfg(fmt: str, algorithm: str):
    if fmt == "fp8":
        if algorithm == "awq_full":
            cfg = mtq.FP8_DEFAULT_CFG.copy()
            cfg["algorithm"] = {"method": "awq_full", "alpha_step": 0.1}
        elif algorithm == "mse":
            cfg = mtq.FP8_DEFAULT_CFG.copy()
            cfg["algorithm"] = "mse"
        else:
            cfg = mtq.FP8_DEFAULT_CFG.copy()
    else:
        if algorithm == "awq_full":
            cfg = mtq.NVFP4_AWQ_FULL_CFG.copy()
        elif algorithm == "mse":
            cfg = mtq.NVFP4_DEFAULT_CFG.copy()
            cfg["algorithm"] = {
                "method": "mse",
                "num_steps": 20,
                "start_multiplier": 0.2,
                "stop_multiplier": 4.5,
            }
        else:
            cfg = mtq.NVFP4_DEFAULT_CFG.copy()

    cfg["quant_cfg"]["*audio_tower*"] = {"enable": False}
    cfg["quant_cfg"]["*lm_head*"]     = {"enable": False}
    return cfg


def main():
    args = parse_args()
    sys.path.insert(0, args.qwen_asr_root)

    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )

    print(f"[CONFIG] format={args.format}  algorithm={args.algorithm}")

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=args.device
    )
    processor = Qwen3ASRProcessor.from_pretrained(
        args.model_path, fix_mistral_regex=True
    )
    device = next(model.parameters()).device

    MULTI_LANGS = [
        "yue_hant_hk", "ar_eg", "de_de", "fr_fr", "es_419",
        "pt_br", "id_id", "it_it", "ko_kr", "ru_ru",
        "th_th", "vi_vn", "ja_jp", "tr_tr", "hi_in",
        "ms_my", "nl_nl", "sv_se", "da_dk", "fi_fi",
        "pl_pl", "cs_cz", "fil_ph", "fa_ir", "el_gr",
        "ro_ro", "hu_hu", "mk_mk",
    ]
    TARGET_TOTAL  = 512
    en_total      = int(TARGET_TOTAL * 0.35)
    zh_total      = int(TARGET_TOTAL * 0.35)
    multi_total   = TARGET_TOTAL - en_total - zh_total

    en_samples    = take_librispeech("train.100", en_total)
    zh_samples    = take_fleurs("cmn_hans_cn", zh_total)
    multi_samples = []
    base_per_lang = multi_total // len(MULTI_LANGS)
    remainder     = multi_total % len(MULTI_LANGS)
    for i, lang in enumerate(MULTI_LANGS):
        n = base_per_lang + (1 if i < remainder else 0)
        multi_samples += take_fleurs(lang, n)

    calib_dataloader = en_samples + zh_samples + multi_samples
    print(f"Total calibration samples: {len(calib_dataloader)}")

    def forward_step(thinker, data):
        inputs = processor(
            audio=data["array"],
            sampling_rate=data["sampling_rate"],
            text=data["text"],
            return_tensors="pt",
        )
        inputs = {
            k: v.to(dtype=torch.bfloat16, device=device) if v.dtype == torch.float32
            else v.to(device=device)
            for k, v in inputs.items()
        }
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return thinker(
            input_ids=inputs["input_ids"],
            input_features=inputs.get("input_features"),
            attention_mask=inputs.get("attention_mask"),
            feature_attention_mask=inputs.get("feature_attention_mask"),
            labels=labels,
        )

    cfg     = build_quant_cfg(args.format, args.algorithm)
    thinker = model.thinker

    def forward_loop(model):
        for data in calib_dataloader:
            forward_step(model, data)

    thinker = mtq.quantize(thinker, cfg, forward_loop)

    log_path = f"quantize_decisions_{args.format}.txt"
    buf = io.StringIO()
    with redirect_stdout(buf):
        mtq.print_quant_summary(thinker)
    with open(log_path, "w") as f:
        f.write(buf.getvalue())
    print(f"[LOG] Quantization decisions: {log_path}")

    mto.save(thinker, args.output_pt)
    print(f"[DONE] Saved: {args.output_pt}")


if __name__ == "__main__":
    main()
