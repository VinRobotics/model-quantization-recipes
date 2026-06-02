#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
qad_train.py — Quantization-Aware Distillation (QAD) for ASR models

Architecture (default: Qwen3-ASR family):
  TeacherModel (full-precision, FROZEN)
  StudentModel (INT4 fake-quant, trainable decoder only)
    └── audio_tower  ← FROZEN  (via frozen_keywords)
    └── lm_head      ← FROZEN  (via frozen_keywords)
    └── model/decoder← TRAINED

Loss (response tokens only, after <audio_end> marker):
  L = alpha_kd * KL(p_T ∥ p_S) + (1 - alpha_kd) * CE(logits_S, pseudo_labels)

WER Eval:
  Runs on VIVOS (Vietnamese) + LibriSpeech (English) after every checkpoint save.
  Results appended to output_dir/wer_log.jsonl; summary table written at end.

Workflow (all 3 stages run automatically via run_qad.sh):
  Stage 1 — Pseudo-label generation (single GPU):
    python qad_train.py --config configs/qad_example.yaml --mode prepare

  Stage 2 — Sanity check (single GPU):
    python qad_train.py --config configs/qad_example.yaml --mode sanity

  Stage 3 — QAD training (multi-GPU DDP):
    torchrun --nproc_per_node=N qad_train.py --config configs/qad_example.yaml --mode train

Generalization notes:
  - Model class names are specified in config (model section), not hardcoded.
  - custom_src is optional; set to null to use standard HuggingFace classes.
  - audio_end_token_id and audio_prompt are auto-detected from model/processor
    or overridden via config.
  - Teacher and student can be any compatible checkpoint (different sizes, variants).
  - eval_model_class must expose a .transcribe(audio, language) -> [Result] interface.
"""

from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import yaml
import torchaudio
import torch.nn.functional as F
import torch.distributed as dist
import torch
import soundfile as sf
import numpy as np
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import transformers

import argparse
import glob
import importlib
import json
import logging
import math
import os
import re
import sys
import time
import warnings
warnings.filterwarnings("ignore")

transformers.logging.set_verbosity_error()


log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathsCfg:
    teacher_ckpt:  str              # Full-precision teacher checkpoint
    student_ckpt:  str              # Quantized student checkpoint (PTQ: INT4, INT8, FP8, …)
    data_dir:      str              # Root directory containing .wav files
    pseudo_labels: str              # Output JSONL for pseudo-label generation
    output_dir:    str              # Directory for checkpoints, logs, summaries
    eval_vivos:    str              # VIVOS eval set: prompts.txt + waves/
    eval_libri:    str              # LibriSpeech eval set: prompts.txt + waves/
    custom_src:    Optional[str] = None  # Custom model source; None = use standard HF


@dataclass
class ModelCfg:
    """
    Controls which classes are imported for teacher/student/eval.

    If custom_src is set in PathsCfg, these classes are imported from that
    directory. Otherwise they are imported from the 'transformers' package
    (or any installed package accessible on PYTHONPATH).

    eval_model_class must expose:
        model.transcribe(audio=(array, sr), language=None) -> [Result]
    where Result has a .text attribute.
    """
    model_class:           str  = "Qwen3ASRForConditionalGeneration"
    processor_class:       str  = "Qwen3ASRProcessor"
    eval_model_class:      str  = "Qwen3ASRModel"
    eval_model_module:     str  = "qwen_asr"   # module to import eval_model_class from
    model_module:          str  = "qwen_asr.core.transformers_backend.modeling_qwen3_asr"
    processor_module:      str  = "qwen_asr.core.transformers_backend.processing_qwen3_asr"

    # Token / prompt overrides (None = auto-detect from model or processor)
    audio_end_token_id:    Optional[int] = None
    audio_prompt_override: Optional[str] = None   # e.g. "<|audio_bos|><|AUDIO|><|audio_eos|>"

    # Processor extra kwargs (e.g. fix_mistral_regex=True for Qwen3-ASR)
    processor_kwargs:      dict = field(default_factory=dict)

    # Quantization backend used to save the student checkpoint.
    # "modelopt" : calls mto.enable_huggingface_checkpointing() before loading
    #              (required for NVIDIA modelopt INT4 / INT8 / FP8 checkpoints)
    # null       : standard HuggingFace from_pretrained() — use for any other
    #              PTQ scheme (bitsandbytes, GPTQ, AWQ, HQQ, etc.)
    quantization_backend:  Optional[str] = "modelopt"

    # Set True to untie lm_head from embed_tokens if they share the same tensor.
    # Commonly required for PTQ-quantized models where tied weights conflict
    # with per-layer quantization while keeping lm_head frozen.
    fix_tie_embeddings:    bool = True

    # Inner module name that wraps the language model (e.g. "thinker" for Qwen3-ASR)
    # Set to null / empty string if the model has no inner wrapper.
    inner_module:          Optional[str] = "thinker"


@dataclass
class DataCfg:
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    sample_rate:    int   = 16000


@dataclass
class GenerationCfg:
    max_new_tokens: int = 256
    num_beams:      int = 1


@dataclass
class TrainingCfg:
    lr:                          float = 5e-6
    min_lr:                      float = 5e-7
    warmup_steps:                int   = 300
    max_steps:                   int   = 45000
    per_gpu_batch_size:          int   = 4
    gradient_accumulation_steps: int   = 2
    weight_decay:                float = 0.01
    grad_clip:                   float = 1.0
    kd_temperature:              float = 2.0
    alpha_kd:                    float = 1.0
    bf16:                        bool  = True
    save_interval:               int   = 1000
    log_interval:                int   = 10
    seed:                        int   = 42
    num_workers:                 int   = 4
    adam_betas:                  list  = field(default_factory=lambda: [0.9, 0.95])
    adam_eps:                    float = 1e-8
    frozen_keywords:             list  = field(default_factory=lambda: ["audio_tower", "lm_head"])


@dataclass
class EvalCfg:
    max_samples_vivos: int = 760
    max_samples_libri: int = 500
    dtype:             str = "float16"   # "float16" or "bfloat16"


@dataclass
class Config:
    paths:      PathsCfg
    model:      ModelCfg      = field(default_factory=ModelCfg)
    data:       DataCfg       = field(default_factory=DataCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    training:   TrainingCfg   = field(default_factory=TrainingCfg)
    eval:       EvalCfg       = field(default_factory=EvalCfg)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Expand ~ in all path values
    paths_raw = {
        k: os.path.expanduser(str(v)) if v is not None else None
        for k, v in raw["paths"].items()
    }

    return Config(
        paths=PathsCfg(**paths_raw),
        model=ModelCfg(**raw.get("model", {})),
        data=DataCfg(**raw.get("data", {})),
        generation=GenerationCfg(**raw.get("generation", {})),
        training=TrainingCfg(**raw.get("training", {})),
        eval=EvalCfg(**raw.get("eval", {})),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return rank() == 0


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    log.info(f"DDP: rank={rank()}/{world_size()} on cuda:{local_rank}")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format=f"[%(asctime)s][rank{rank()}][%(levelname)s] %(message)s",
        level=level if is_main() else logging.WARNING,
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic class import
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_custom_src(custom_src: Optional[str]):
    """Insert custom_src at the front of sys.path if set and not already present."""
    if custom_src is None:
        return
    src = os.path.expanduser(custom_src)
    if src not in sys.path:
        sys.path.insert(0, src)


def _import_class(module_name: str, class_name: str) -> type:
    """Import a class by module + class name string."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot import '{class_name}' from '{module_name}': {e}\n"
            f"Check model.model_module / processor_module / eval_model_module in your config."
        ) from e


def get_model_class(cfg: Config) -> type:
    _ensure_custom_src(cfg.paths.custom_src)
    return _import_class(cfg.model.model_module, cfg.model.model_class)


def get_processor_class(cfg: Config) -> type:
    _ensure_custom_src(cfg.paths.custom_src)
    return _import_class(cfg.model.processor_module, cfg.model.processor_class)


def get_eval_model_class(cfg: Config) -> type:
    _ensure_custom_src(cfg.paths.custom_src)
    return _import_class(cfg.model.eval_model_module, cfg.model.eval_model_class)


# ─────────────────────────────────────────────────────────────────────────────
# Model / processor utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_inner_module(model: torch.nn.Module, inner_name: Optional[str]) -> torch.nn.Module:
    """
    Return the inner language-model sub-module (e.g. model.thinker).
    Returns model itself if inner_name is None or empty.
    """
    if not inner_name:
        return model
    if hasattr(model, inner_name):
        return getattr(model, inner_name)
    raise AttributeError(
        f"Model has no attribute '{inner_name}'. "
        f"Set model.inner_module to null in config if there is no wrapper module."
    )


def get_audio_end_token_id(model: torch.nn.Module, cfg: Config) -> int:
    """
    Resolve audio_end_token_id: config override → model.thinker.config → model.config.
    Raises ValueError if not found.
    """
    if cfg.model.audio_end_token_id is not None:
        return cfg.model.audio_end_token_id

    inner = get_inner_module(model, cfg.model.inner_module)
    inner_cfg = getattr(inner, "config", None)
    if inner_cfg and hasattr(inner_cfg, "audio_end_token_id"):
        return inner_cfg.audio_end_token_id

    top_cfg = getattr(model, "config", None)
    if top_cfg and hasattr(top_cfg, "audio_end_token_id"):
        return top_cfg.audio_end_token_id

    raise ValueError(
        "Cannot auto-detect audio_end_token_id from model config. "
        "Set model.audio_end_token_id explicitly in your config YAML."
    )


def get_vocab_size(model: torch.nn.Module, cfg: Config) -> int:
    """Resolve vocab size from inner module or top-level model config."""
    inner = get_inner_module(model, cfg.model.inner_module)
    if hasattr(inner, "vocab_size"):
        return inner.vocab_size
    inner_cfg = getattr(inner, "config", None)
    if inner_cfg and hasattr(inner_cfg, "vocab_size"):
        return inner_cfg.vocab_size
    top_cfg = getattr(model, "config", None)
    if top_cfg and hasattr(top_cfg, "vocab_size"):
        return top_cfg.vocab_size
    raise ValueError("Cannot resolve vocab_size from model. Check your model architecture.")


def get_audio_prompt(processor: Any, cfg: Config) -> str:
    """
    Build the audio prompt string.
    Priority: config override → auto-detect from processor token attributes.
    """
    if cfg.model.audio_prompt_override:
        return cfg.model.audio_prompt_override

    bos = getattr(processor, "audio_bos_token", "")
    tok = getattr(processor, "audio_token", "")
    eos = getattr(processor, "audio_eos_token", "")
    prompt = f"{bos}{tok}{eos}"
    if not prompt.strip():
        raise ValueError(
            "Cannot auto-detect audio_prompt from processor attributes "
            "(audio_bos_token, audio_token, audio_eos_token). "
            "Set model.audio_prompt_override in your config YAML."
        )
    return prompt


def forward_inner(model: torch.nn.Module, cfg: Config, inputs: dict) -> torch.Tensor:
    """Run the inner language model forward pass and return logits."""
    inner = get_inner_module(model, cfg.model.inner_module)
    return inner(**inputs).logits


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher(cfg: Config, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    ModelClass = get_model_class(cfg)
    log.info(f"Loading teacher [{cfg.model.model_class}]: {cfg.paths.teacher_ckpt}")
    teacher = ModelClass.from_pretrained(
        cfg.paths.teacher_ckpt, torch_dtype=dtype, device_map=None,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_student(cfg: Config, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    backend = (cfg.model.quantization_backend or "").lower().strip()

    if backend == "modelopt":
        try:
            import modelopt.torch.opt as mto
            mto.enable_huggingface_checkpointing()
            log.info("modelopt HuggingFace checkpointing enabled")
        except ImportError as e:
            raise ImportError(
                "quantization_backend is 'modelopt' but the package is not installed. "
                "Install with: pip install nvidia-modelopt\n"
                "Or set model.quantization_backend: null to load with standard HuggingFace."
            ) from e
    elif backend:
        log.warning(
            f"Unknown quantization_backend '{backend}' — loading student with standard "
            "HuggingFace from_pretrained(). Supported values: 'modelopt', null."
        )

    ModelClass = get_model_class(cfg)
    quant_label = f"[{backend}]" if backend else "[standard HF]"
    log.info(
        f"Loading student {quant_label} [{cfg.model.model_class}]: {cfg.paths.student_ckpt}"
    )
    student = ModelClass.from_pretrained(
        cfg.paths.student_ckpt, torch_dtype=dtype, device_map=None,
    ).to(device)

    if cfg.model.fix_tie_embeddings:
        _fix_tied_embeddings(student, cfg)

    return student


def _fix_tied_embeddings(model: torch.nn.Module, cfg: Config):
    """
    Untie lm_head from embed_tokens if they share the same data pointer.
    Commonly required for PTQ-quantized models where tied weights conflict
    with per-layer quantization while keeping lm_head frozen.
    """
    inner = get_inner_module(model, cfg.model.inner_module)
    lm_head      = getattr(inner, "lm_head", None)
    embed_tokens  = getattr(getattr(inner, "model", inner), "embed_tokens", None)

    if lm_head is None or embed_tokens is None:
        log.debug("fix_tie_embeddings: lm_head or embed_tokens not found — skipping")
        return

    if lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        inner_cfg = getattr(inner, "config", None)
        if inner_cfg is not None:
            inner_cfg.tie_word_embeddings = False
        log.info("lm_head untied from embed_tokens; tie_word_embeddings=False")


def load_processor(cfg: Config) -> Any:
    ProcessorClass = get_processor_class(cfg)
    return ProcessorClass.from_pretrained(
        cfg.paths.teacher_ckpt, **cfg.model.processor_kwargs
    )


def load_eval_model(cfg: Config, ckpt_dir: str, device: torch.device) -> Any:
    """Load the high-level eval model that exposes .transcribe()."""
    EvalClass = get_eval_model_class(cfg)
    eval_dtype = (
        torch.bfloat16 if cfg.eval.dtype == "bfloat16" else torch.float16
    )
    log.info(f"Loading eval model [{cfg.model.eval_model_class}] from {ckpt_dir}")
    return EvalClass.from_pretrained(
        ckpt_dir,
        dtype=eval_dtype,
        device_map=str(device),
        max_new_tokens=cfg.generation.max_new_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str, target_sr: int) -> torch.Tensor:
    wav_np, sr = sf.read(path)
    if len(wav_np.shape) > 1:
        wav_np = np.mean(wav_np, axis=1)
    wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav, sr, target_sr, resampling_method="sinc_interp_kaiser"
        )
    return wav.squeeze(0)


def scan_audio_files(data_dir: str, dcfg: DataCfg) -> list[str]:
    all_wavs = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    valid, skipped = [], 0
    for p in all_wavs:
        try:
            info = sf.info(p)
            dur  = info.frames / info.samplerate
            if dcfg.min_duration_s <= dur <= dcfg.max_duration_s:
                valid.append(p)
            else:
                skipped += 1
        except Exception as e:
            log.warning(f"Cannot read {p}: {e}")
            skipped += 1
    log.info(f"Audio scan: {len(all_wavs)} total → {len(valid)} valid, {skipped} skipped")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class QADDataset(Dataset):
    def __init__(self, jsonl_path: str, dcfg: DataCfg):
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.records = [
            r for r in records
            if os.path.isfile(r["path"]) and r.get("pseudo_text", "").strip()
        ]
        self.sample_rate = dcfg.sample_rate
        log.info(f"QADDataset: {len(self.records)}/{len(records)} valid samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav = load_wav(rec["path"], self.sample_rate)
        return {
            "path":        rec["path"],
            "array":       wav.numpy(),
            "sr":          self.sample_rate,
            "pseudo_text": rec["pseudo_text"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs_batch(
    processor,
    samples:      list[dict],
    device:       torch.device,
    dtype:        torch.dtype,
    pad_token_id: int,
    audio_prompt: str,
) -> dict[str, torch.Tensor]:
    """
    Build a padded batch from a list of sample dicts.
    Each sample must have: array (np.ndarray), sr (int), pseudo_text (str).

    Tensor shapes after processor:
      - 2D tensors (e.g. input_ids, attention_mask): [1, seq_len] → pad to max_seq_len
      - 3D tensors (e.g. audio features): [1, channels, frames] → pad to max_frames
      - Other tensors: concatenated on dim 0
    """
    per_sample = []
    for s in samples:
        full_text = audio_prompt + s["pseudo_text"]
        inputs = processor(
            text=full_text,
            audio=s["array"],
            sampling_rate=s["sr"],
            return_tensors="pt",
        )
        per_sample.append(inputs)

    out: dict[str, torch.Tensor] = {}
    for k in per_sample[0].keys():
        tensors = [ps[k] for ps in per_sample]
        t0 = tensors[0]

        if t0.dim() == 2:
            max_len = max(t.shape[1] for t in tensors)
            pad_val = pad_token_id if k == "input_ids" else 0
            padded  = torch.full((len(tensors), max_len), pad_val, dtype=t0.dtype)
            for i, t in enumerate(tensors):
                padded[i, : t.shape[1]] = t[0]
            out[k] = padded

        elif t0.dim() == 3:
            max_frames = max(t.shape[2] for t in tensors)
            padded = torch.zeros(len(tensors), t0.shape[1], max_frames, dtype=t0.dtype)
            for i, t in enumerate(tensors):
                padded[i, :, : t.shape[2]] = t[0]
            out[k] = padded

        else:
            # Scalar / 1D tensors — concatenate along dim 0
            out[k] = torch.cat(tensors, dim=0)

    # Move to device with correct dtype
    return {
        k: (
            v.to(device=device, dtype=dtype)
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else v.to(device)
        )
        for k, v in out.items()
    }


def build_response_mask_and_labels(
    inputs:             dict,
    audio_end_token_id: int,
    pad_token_id:       int,
    device:             torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a boolean mask over response tokens (everything after <audio_end>)
    and a labels tensor with -100 at non-response positions.
    """
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask")
    B, T      = input_ids.shape

    resp_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    labels    = torch.full((B, T), -100, dtype=torch.long, device=device)

    for b in range(B):
        positions = (input_ids[b] == audio_end_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            log.debug(
                f"sample {b}: audio_end_token_id ({audio_end_token_id}) not found "
                "— excluded from loss"
            )
            continue

        pos_end     = positions[-1].item()
        start_label = pos_end + 1

        if attn_mask is not None:
            resp_mask[b, pos_end:] = attn_mask[b, pos_end:].bool()
        else:
            resp_mask[b, pos_end:] = True

        if start_label < T:
            valid = resp_mask[b, start_label:]
            labels[b, start_label:] = torch.where(
                valid,
                input_ids[b, start_label:],
                torch.full_like(input_ids[b, start_label:], -100),
            )

    return resp_mask, labels


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits_s:   torch.Tensor,
    logits_t:   torch.Tensor,
    labels:     torch.Tensor,
    resp_mask:  torch.Tensor,
    tau:        float,
    alpha_kd:   float,
    vocab_size: int,
) -> Optional[torch.Tensor]:
    """
    Compute combined KD + CE loss on response tokens only.
    Returns None if resp_mask has no active positions.

    Args:
        logits_s:  Student logits  [B, T, V]
        logits_t:  Teacher logits  [B, T, V]
        labels:    Ground-truth token ids, -100 at non-label positions [B, T]
        resp_mask: Boolean mask of response positions [B, T]
        tau:       KD temperature
        alpha_kd:  Weight on KD loss (1.0 = pure KD, 0.0 = pure CE)
        vocab_size: Vocabulary size V
    """
    if not resp_mask.any():
        return None

    ls      = logits_s[resp_mask]
    lt      = logits_t[resp_mask]
    log_p_s = F.log_softmax(ls / tau, dim=-1)
    p_t     = F.softmax(lt  / tau, dim=-1).detach()
    kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)

    if alpha_kd >= 1.0:
        return kd_loss

    shift_logits = logits_s[:, :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    if not (shift_labels != -100).any():
        return kd_loss

    ce_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return alpha_kd * kd_loss + (1.0 - alpha_kd) * ce_loss


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr_lambda(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, max_steps - warmup)
    return (min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (lr - min_lr)) / lr


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / restore
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    student,
    processor,
    optimizer,
    scheduler,
    step:       int,
    output_dir: str,
    tag:        str = "",
) -> str:
    name     = f"step_{step:07d}" + (f"_{tag}" if tag else "")
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    raw = student.module if hasattr(student, "module") else student
    raw.save_pretrained(ckpt_dir)

    if processor is not None:
        try:
            processor.save_pretrained(ckpt_dir)
        except Exception as e:
            log.warning(f"processor.save_pretrained failed (non-critical): {e}")

    torch.save(
        {
            "step":      step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    log.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    states = sorted(glob.glob(os.path.join(output_dir, "step_*", "trainer_state.pt")))
    return os.path.dirname(states[-1]) if states else None


def restore_student_weights(student: torch.nn.Module, ckpt_dir: str, device: torch.device):
    from safetensors.torch import load_file

    raw        = student.module if hasattr(student, "module") else student
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state: dict = {}
        for shard in shard_files:
            state.update(load_file(os.path.join(ckpt_dir, shard), device=str(device)))
        log.info(f"Restored weights from {len(shard_files)} shards in {ckpt_dir}")
    else:
        single = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.isfile(single):
            log.warning(f"No safetensors found in {ckpt_dir} — skipping weight restore")
            return
        state = load_file(single, device=str(device))
        log.info(f"Restored weights from {single}")

    missing, unexpected = raw.load_state_dict(state, strict=False)
    if missing:
        log.warning(
            f"Missing keys ({len(missing)}): {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        log.warning(
            f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WER Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase + strip punctuation for WER comparison."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def load_eval_dataset(data_dir: str, max_samples: int) -> list[tuple[str, str]]:
    """
    Load (wav_path, reference_text) pairs from an eval dataset folder.

    Expected layout:
        data_dir/
            prompts.txt     # lines: UTT_ID TRANSCRIPT
            waves/
                UTT_ID.wav  # flat or nested
    """
    prompts_path = os.path.join(data_dir, "prompts.txt")
    waves_dir    = os.path.join(data_dir, "waves")

    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompts.txt not found: {prompts_path}")
    if not os.path.isdir(waves_dir):
        raise FileNotFoundError(f"waves/ dir not found: {waves_dir}")

    pairs: list[tuple[str, str]] = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, ref_text = parts

            wav_path = os.path.join(waves_dir, f"{utt_id}.wav")
            if not os.path.isfile(wav_path):
                matches = glob.glob(
                    os.path.join(waves_dir, "**", f"{utt_id}.wav"), recursive=True
                )
                if not matches:
                    continue
                wav_path = matches[0]
            pairs.append((wav_path, ref_text))

    if max_samples > 0:
        pairs = pairs[:max_samples]
    return pairs


@torch.no_grad()
def run_wer_eval_dataset(
    eval_model,
    data_dir:     str,
    max_samples:  int,
    dataset_name: str,
) -> float:
    """
    Compute WER on one eval dataset.
    eval_model must expose: .transcribe(audio=(array, sr), language=None) -> [Result]
    Returns WER (float) or -1.0 on error.
    """
    try:
        from jiwer import wer as compute_wer
    except ImportError:
        log.warning("jiwer not installed — skipping WER eval. Install with: pip install jiwer")
        return -1.0

    try:
        pairs = load_eval_dataset(data_dir, max_samples)
    except Exception as e:
        log.warning(f"[{dataset_name}] Cannot load eval data: {e}")
        return -1.0

    if not pairs:
        log.warning(f"[{dataset_name}] No valid pairs found in {data_dir}")
        return -1.0

    hypotheses, references = [], []
    errors = 0
    t0 = time.time()

    for i, (wav_path, ref_text) in enumerate(pairs):
        try:
            audio_array, sample_rate = sf.read(wav_path)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)

            results  = eval_model.transcribe(audio=(audio_array, sample_rate), language=None)
            hyp_text = results[0].text
        except Exception as e:
            if i == 0:
                import traceback
                log.warning(f"[{dataset_name}] sample {i} failed: {e}\n{traceback.format_exc()}")
            else:
                log.warning(f"[{dataset_name}] sample {i} failed: {e}")
            hyp_text = ""
            errors  += 1

        hypotheses.append(normalize_text(hyp_text))
        references.append(normalize_text(ref_text))

    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not valid_pairs:
        log.warning(f"[{dataset_name}] No valid reference-hypothesis pairs")
        return -1.0

    refs, hyps = zip(*valid_pairs)
    wer_score  = compute_wer(list(refs), list(hyps))
    elapsed    = time.time() - t0

    log.info(
        f"  [{dataset_name}] WER={wer_score:.4f} ({wer_score * 100:.2f}%) "
        f"| {len(valid_pairs)} samples | {errors} errors | {elapsed:.1f}s"
    )
    return wer_score


def run_wer_eval(cfg: Config, eval_model: Any, step: int) -> dict:
    """Run WER on VIVOS + LibriSpeech and return result dict."""
    results = {"step": step, "timestamp": datetime.now().isoformat()}

    log.info("[WER Eval] Evaluating VIVOS...")
    results["wer_vivos"] = run_wer_eval_dataset(
        eval_model, cfg.paths.eval_vivos, cfg.eval.max_samples_vivos, "VIVOS"
    )

    log.info("[WER Eval] Evaluating LibriSpeech...")
    results["wer_libri"] = run_wer_eval_dataset(
        eval_model, cfg.paths.eval_libri, cfg.eval.max_samples_libri, "LibriSpeech"
    )

    log.info(
        f"[WER Eval] step={step} "
        f"VIVOS={results['wer_vivos']:.4f}  "
        f"LibriSpeech={results['wer_libri']:.4f}"
    )
    return results


def append_wer_log(output_dir: str, result: dict):
    log_path = os.path.join(output_dir, "wer_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_wer_summary(output_dir: str):
    """Read wer_log.jsonl and write a formatted summary table to wer_summary.txt."""
    log_path     = os.path.join(output_dir, "wer_log.jsonl")
    summary_path = os.path.join(output_dir, "wer_summary.txt")

    if not os.path.isfile(log_path):
        log.warning("wer_log.jsonl not found — skipping summary")
        return

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return

    col_step  = 10
    col_vivos = 14
    col_libri = 18
    col_ts    = 26
    sep = "-" * (col_step + col_vivos + col_libri + col_ts + 13)

    best_vivos = min(
        (r for r in records if r.get("wer_vivos", -1) >= 0),
        key=lambda r: r["wer_vivos"], default=None,
    )
    best_libri = min(
        (r for r in records if r.get("wer_libri", -1) >= 0),
        key=lambda r: r["wer_libri"], default=None,
    )

    lines = [
        "=" * len(sep),
        "  QAD WER Evaluation Summary",
        f"  Output dir: {output_dir}",
        f"  Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * len(sep),
        (
            f"{'Step':>{col_step}} | {'VIVOS WER':>{col_vivos}} | "
            f"{'LibriSpeech WER':>{col_libri}} | {'Timestamp':<{col_ts}}"
        ),
        sep,
    ]

    for r in records:
        vivos_str  = f"{r['wer_vivos']:.4f}" if r.get("wer_vivos", -1) >= 0 else "N/A"
        libri_str  = f"{r['wer_libri']:.4f}" if r.get("wer_libri", -1) >= 0 else "N/A"
        ts_str     = r.get("timestamp", "")[:19]
        vivos_mark = " ★" if best_vivos and r["step"] == best_vivos["step"] else ""
        libri_mark = " ★" if best_libri and r["step"] == best_libri["step"] else ""
        lines.append(
            f"{r['step']:>{col_step}} | {vivos_str + vivos_mark:>{col_vivos}} | "
            f"{libri_str + libri_mark:>{col_libri}} | {ts_str:<{col_ts}}"
        )

    lines.append(sep)
    if best_vivos:
        lines.append(f"  Best VIVOS WER      : {best_vivos['wer_vivos']:.4f} at step {best_vivos['step']}")
    if best_libri:
        lines.append(f"  Best LibriSpeech WER: {best_libri['wer_libri']:.4f} at step {best_libri['step']}")
    lines.append("=" * len(sep))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"WER summary written → {summary_path}")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Pseudo-label generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_prepare(cfg: Config, batch_size: int = 8):
    """
    Stage 1: Generate pseudo transcripts with the teacher model.
    Uses eval_model_class which must expose .transcribe().
    """
    device = torch.device("cuda:0")
    torch.bfloat16 if cfg.training.bf16 else torch.float16

    log.info(f"Loading teacher for pseudo-label generation: {cfg.paths.teacher_ckpt}")
    teacher = load_eval_model(cfg, cfg.paths.teacher_ckpt, device)

    wav_files = scan_audio_files(cfg.paths.data_dir, cfg.data)
    out_path  = os.path.expanduser(cfg.paths.pseudo_labels)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Resume from partial run
    done: set[str] = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["path"])
        log.info(f"Resuming: {len(done)} already done")

    remaining   = [p for p in wav_files if p not in done]
    empty_count = 0
    log.info(f"Generating pseudo labels for {len(remaining)} files → {out_path}")

    with open(out_path, "a") as fout:
        for i in range(0, len(remaining), batch_size):
            batch_paths = remaining[i : i + batch_size]
            try:
                batch_inputs = []
                for path in batch_paths:
                    wav_np, sr = sf.read(path)
                    if len(wav_np.shape) > 1:
                        wav_np = np.mean(wav_np, axis=1)
                    batch_inputs.append((wav_np.astype(np.float32), sr))

                results = [
                    teacher.transcribe(audio=inp, language=None)
                    for inp in batch_inputs
                ]

                for path, result in zip(batch_paths, results):
                    pseudo = result[0].text.strip()
                    if not pseudo:
                        empty_count += 1
                    fout.write(
                        json.dumps({"path": path, "pseudo_text": pseudo}, ensure_ascii=False)
                        + "\n"
                    )
                fout.flush()

            except Exception as e:
                log.warning(f"Failed batch starting {batch_paths[0]}: {e}")

            if (i + batch_size) % (100 * batch_size) == 0:
                log.info(f"Prepare: {i + batch_size}/{len(remaining)} | empty={empty_count}")

    log.info(f"Pseudo-label generation complete. Total empty: {empty_count}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check(cfg: Config, n_steps: int = 5):
    """
    Stage 2: Overfit 4 samples for n_steps.
    Verifies: model loading, input pipeline, loss computation, backward pass.
    """
    log.info("=== Sanity Check ===")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype     = torch.bfloat16 if cfg.training.bf16 else torch.float16
    teacher   = load_teacher(cfg, device, dtype)
    student   = load_student(cfg, device, dtype)
    processor = load_processor(cfg)

    audio_end_token_id = get_audio_end_token_id(student, cfg)
    pad_token_id       = processor.tokenizer.pad_token_id
    vocab_size         = get_vocab_size(student, cfg)
    audio_prompt       = get_audio_prompt(processor, cfg)

    log.info(
        f"audio_end_token_id={audio_end_token_id} "
        f"pad_token_id={pad_token_id} "
        f"vocab_size={vocab_size}"
    )

    for name, param in student.named_parameters():
        if any(k in name for k in cfg.training.frozen_keywords):
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=cfg.training.lr
    )

    dataset = QADDataset(os.path.expanduser(cfg.paths.pseudo_labels), cfg.data)
    samples = [dataset[i] for i in range(min(4, len(dataset)))]

    student.train()
    losses = []
    for step in range(1, n_steps + 1):
        inputs            = make_inputs_batch(
            processor, samples, device, dtype, pad_token_id, audio_prompt
        )
        resp_mask, labels = build_response_mask_and_labels(
            inputs, audio_end_token_id, pad_token_id, device
        )
        log.info(f"  step {step}: resp_mask active tokens = {resp_mask.sum().item()}")

        if not resp_mask.any():
            raise RuntimeError(
                "resp_mask is empty — check audio_end_token_id or processor format"
            )

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits_t = forward_inner(teacher, cfg, inputs).detach()

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits_s = forward_inner(student, cfg, inputs)

        loss = compute_loss(
            logits_s, logits_t, labels, resp_mask,
            cfg.training.kd_temperature, cfg.training.alpha_kd, vocab_size,
        )
        if loss is None:
            raise RuntimeError("compute_loss returned None — no valid response tokens")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        log.info(f"  step {step}/{n_steps}: loss={loss.item():.4f}")

    if not all(math.isfinite(v) for v in losses):
        raise RuntimeError(f"Non-finite loss detected: {losses}")

    log.info(f"PASSED — losses: {[f'{v:.4f}' for v in losses]}")
    log.info("=== Sanity Check Done ===")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: QAD Training
# ─────────────────────────────────────────────────────────────────────────────

def run_train(cfg: Config):
    setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}")
    dtype      = torch.bfloat16 if cfg.training.bf16 else torch.float16
    output_dir = os.path.expanduser(cfg.paths.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    writer = (
        SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        if is_main() else None
    )
    torch.manual_seed(cfg.training.seed + rank())

    teacher   = load_teacher(cfg, device, dtype)
    student   = load_student(cfg, device, dtype)
    processor = load_processor(cfg)

    audio_end_token_id = get_audio_end_token_id(student, cfg)
    pad_token_id       = processor.tokenizer.pad_token_id
    vocab_size         = get_vocab_size(student, cfg)
    audio_prompt       = get_audio_prompt(processor, cfg)

    log.info(
        f"audio_end_token_id={audio_end_token_id} "
        f"pad_token_id={pad_token_id} "
        f"vocab_size={vocab_size}"
    )

    # Freeze specified modules
    for name, param in student.named_parameters():
        if any(k in name for k in cfg.training.frozen_keywords):
            param.requires_grad_(False)

    # Gradient checkpointing (optional, saves memory)
    inner = get_inner_module(student, cfg.model.inner_module)
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    if is_distributed():
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[local_rank], find_unused_parameters=False
        )

    tcfg    = cfg.training
    dataset = QADDataset(os.path.expanduser(cfg.paths.pseudo_labels), cfg.data)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    loader  = DataLoader(
        dataset,
        batch_size=tcfg.per_gpu_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=tcfg.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=tcfg.lr,
        betas=tuple(tcfg.adam_betas),
        weight_decay=tcfg.weight_decay,
        eps=tcfg.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(
            s, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr
        ),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    global_step       = 0
    valid_micro_steps = 0

    latest = find_latest_checkpoint(output_dir)
    if latest:
        log.info(f"Resuming from {latest}")
        state       = torch.load(os.path.join(latest, "trainer_state.pt"), map_location="cpu")
        global_step = state["step"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_student_weights(student, latest, device)
        valid_micro_steps = global_step * tcfg.gradient_accumulation_steps
        log.info(f"Resumed at step {global_step}")

    accum_steps  = tcfg.gradient_accumulation_steps
    running_loss = 0.0
    optimizer.zero_grad()

    log.info(
        f"QAD training start | max_steps={tcfg.max_steps} "
        f"GBS={tcfg.per_gpu_batch_size * accum_steps * world_size()} "
        f"τ={tcfg.kd_temperature} α_kd={tcfg.alpha_kd} lr={tcfg.lr}"
    )

    epoch = 0
    while global_step < tcfg.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for samples in loader:
            if global_step >= tcfg.max_steps:
                break

            # ── Validate batch across all ranks ─────────────────────────────
            batch_ok = torch.tensor(1, device=device)
            try:
                inputs            = make_inputs_batch(
                    processor, samples, device, dtype, pad_token_id, audio_prompt
                )
                resp_mask, labels = build_response_mask_and_labels(
                    inputs, audio_end_token_id, pad_token_id, device
                )
                if not resp_mask.any():
                    if is_main():
                        log.warning("No response tokens in batch — skipping")
                    batch_ok.fill_(0)
            except Exception as e:
                log.warning(f"Batch build failed rank={rank()}: {e}")
                batch_ok.fill_(0)

            if is_distributed():
                dist.all_reduce(batch_ok, op=dist.ReduceOp.MIN)
            if batch_ok.item() == 0:
                continue

            # ── Teacher forward (no grad) ────────────────────────────────────
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits_t = forward_inner(teacher, cfg, inputs).detach()

            # ── Student forward ──────────────────────────────────────────────
            # Bypass DDP wrapper for gradient accumulation; gradients are
            # manually all-reduced below at the optimizer step boundary.
            raw_student = student.module if hasattr(student, "module") else student
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits_s = forward_inner(raw_student, cfg, inputs)

            # ── Loss ────────────────────────────────────────────────────────
            loss = compute_loss(
                logits_s, logits_t, labels, resp_mask,
                tcfg.kd_temperature, tcfg.alpha_kd, vocab_size,
            )
            if loss is None:
                continue

            (loss / accum_steps).backward()

            loss_log = loss.detach().clone()
            if is_distributed():
                dist.all_reduce(loss_log, op=dist.ReduceOp.AVG)
            running_loss += loss_log.item() / accum_steps

            valid_micro_steps += 1

            if valid_micro_steps % accum_steps == 0:
                # Manual gradient sync (since we bypassed DDP wrapper)
                if is_distributed():
                    for param in student.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    tcfg.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ─────────────────────────────────────────────────
                if is_main() and global_step % tcfg.log_interval == 0:
                    avg_loss  = running_loss / tcfg.log_interval
                    lr_now    = scheduler.get_last_lr()[0]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad],
                        float("inf"),
                    ).item()
                    log.info(
                        f"step={global_step:>6d}/{tcfg.max_steps}  "
                        f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                        f"grad_norm={grad_norm:.3f}"
                    )
                    if writer:
                        writer.add_scalar("train/loss",      avg_loss,  global_step)
                        writer.add_scalar("train/lr",        lr_now,    global_step)
                        writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    running_loss = 0.0

                # ── Checkpoint + WER eval ────────────────────────────────────
                if global_step % tcfg.save_interval == 0:
                    if is_main():
                        ckpt_dir = save_checkpoint(
                            student, processor, optimizer, scheduler,
                            global_step, output_dir,
                        )
                        eval_model = load_eval_model(cfg, ckpt_dir, device)
                        wer_result = run_wer_eval(cfg, eval_model, global_step)
                        del eval_model
                        torch.cuda.empty_cache()
                        append_wer_log(output_dir, wer_result)
                        if writer:
                            if wer_result["wer_vivos"] >= 0:
                                writer.add_scalar("eval/wer_vivos",       wer_result["wer_vivos"], global_step)
                            if wer_result["wer_libri"] >= 0:
                                writer.add_scalar("eval/wer_librispeech", wer_result["wer_libri"], global_step)
                    if is_distributed():
                        dist.barrier()

    # ── Final checkpoint ─────────────────────────────────────────────────────
    if is_main():
        ckpt_dir = save_checkpoint(
            student, processor, optimizer, scheduler,
            global_step, output_dir, tag="final",
        )
        eval_model = load_eval_model(cfg, ckpt_dir, device)
        wer_result = run_wer_eval(cfg, eval_model, global_step)
        del eval_model
        torch.cuda.empty_cache()
        append_wer_log(output_dir, wer_result)
        if writer:
            if wer_result["wer_vivos"] >= 0:
                writer.add_scalar("eval/wer_vivos",       wer_result["wer_vivos"], global_step)
            if wer_result["wer_libri"] >= 0:
                writer.add_scalar("eval/wer_librispeech", wer_result["wer_libri"], global_step)
        write_wer_summary(output_dir)
        log.info(f"Training complete at step {global_step}")

    if writer:
        writer.close()

    if is_distributed():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantization-Aware Distillation (QAD) for ASR models"
    )
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--mode",    required=True, choices=["prepare", "sanity", "train"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    if args.mode == "prepare":
        run_prepare(cfg)
    elif args.mode == "sanity":
        run_sanity_check(cfg)
    else:
        run_train(cfg)


if __name__ == "__main__":
    main()
