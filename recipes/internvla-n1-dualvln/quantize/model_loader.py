# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Model loading, calibration forward loops, and export for Qwen2.5-VL.

Logic mirrors NVIDIA's reference quantize_and_export implementation, with the
transformers >= 5.x workaround patches kept intact.

There is deliberately no InternVLA-N1 branch here. ``repackage_system2.py`` runs first
and turns the checkpoint into a stock Qwen2.5-VL, so this module never sees an
``internvla_n1`` config and never needs to import InternNav. Loading the original
checkpoint directly (with ``config.system1 = "none"`` to suppress System 1) also works
and saves the intermediate copy, but it puts a gated third-party repository -- and the
three monkeypatches needed to load it -- on the critical path of every quantization run.
"""

import os
import shutil
from typing import Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(
    model_dir: str,
    dtype: str = "bf16",
    device: str = "cuda",
):
    """Load model + tokenizer + optional processor via Auto* classes.

    Mirrors NVIDIA's ``_load_model`` for the standard VLM path: tries
    ``AutoModelForImageTextToText`` first (so multimodal architectures are
    not silently downgraded to their text-only registration), then falls
    back to ``AutoModelForCausalLM`` and finally ``AutoModel``.
    """
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True
    )

    try:
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            min_pixels=128 * 28 * 28,
            max_pixels=2048 * 32 * 32,
        )
    except Exception:
        processor = None

    last_err: Optional[Exception] = None
    model = None
    for factory in (
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoModel,
    ):
        try:
            model = factory.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)
            break
        except (ValueError, KeyError) as e:
            last_err = e
    if model is None:
        raise RuntimeError(
            f"Could not load {model_dir} via any AutoModel factory"
        ) from last_err

    model.to(torch_dtype)

    # ModelOpt export_hf_checkpoint crashes when architectures is None.
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor


# --------------------------------------------------------------------------- #
# Calibration forward loops
# --------------------------------------------------------------------------- #
def calibrate_text(model, dataloader):
    """Forward-loop calibration pass for text-only DataLoader.

    Mirrors NVIDIA's ``_calibrate``.
    """
    for data in tqdm(dataloader, desc="Calibrating (text)"):
        data = data.to(model.device)
        model(data)


def calibrate_multimodal(model, batches):
    """Forward-loop calibration pass for multimodal ``BatchFeature`` dicts.

    Mirrors NVIDIA's ``_calibrate_multimodal``. Float inputs (pixel_values)
    inherit the model's dtype; int inputs (input_ids, attention_mask) are
    untouched. ``use_cache=False`` matches the reference.

    NaN amax assertions from ModelOpt's internal calibrator are caught and
    the offending batch is skipped so calibration can complete on the
    remaining valid samples.
    """
    device = model.device
    model_dtype = next(model.parameters()).dtype
    valid_batches = 0
    skipped_nan_batches = 0

    for batch in tqdm(batches, desc="Calibrating (multimodal)"):
        kwargs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if v.dtype.is_floating_point:
                    v = v.to(model_dtype)
            kwargs[k] = v
        kwargs.setdefault("use_cache", False)

        with torch.no_grad():
            try:
                model(**kwargs)
                valid_batches += 1
            except AssertionError as exc:
                if "detected nan values in amax" in str(exc):
                    skipped_nan_batches += 1
                    continue
                raise

    if valid_batches == 0:
        raise RuntimeError(
            "All multimodal calibration batches were skipped due to NaN amax."
        )
    if skipped_nan_batches > 0:
        print(
            f"[WAR] Skipped {skipped_nan_batches} multimodal calibration "
            f"batch(es) with NaN amax."
        )


# --------------------------------------------------------------------------- #
# WAR patches (transformers >= 5.x)
# --------------------------------------------------------------------------- #
def normalize_tied_weights_keys(model) -> None:
    """WAR for transformers >= 5.x ``_tied_weights_keys`` format change.

    Mirrors NVIDIA's ``_normalize_tied_weights_keys``. Newer transformers
    expects each submodule's ``_tied_weights_keys`` attribute to be a
    dict-like (so ``modeling_utils._get_tied_weight_keys`` can call ``.keys()``).
    Older custom modeling code still declares it as a list, which crashes
    ``model.save_pretrained``.

    Convert list-shaped attributes to ``{key: key}`` dicts in place. The dict's
    keys exactly match the original list, preserving behavior for downstream
    tied-weight tracking. No-op for modules already in the dict format.
    """
    for module in model.modules():
        attr = getattr(module, "_tied_weights_keys", None)
        if isinstance(attr, list):
            module._tied_weights_keys = {k: k for k in attr}


def fix_generation_config_for_strict_validate(model) -> None:
    """WAR for transformers >= 5.x ``GenerationConfig.validate(strict=True)``.

    Mirrors NVIDIA's ``_fix_generation_config_for_strict_validate``. ModelOpt's
    ``export_hf_checkpoint`` calls ``model.save_pretrained`` which runs
    ``validate(strict=True)`` on the generation config; that validator rejects
    HF checkpoints whose ``generation_config.json`` sets sampling-only kwargs
    (``top_p`` / ``top_k`` / ``temperature``) without ``do_sample = True``.

    Force ``do_sample = True`` when any sampling kwarg is present. This only
    changes the saved ``generation_config.json``; runtime params are read
    separately and are not affected.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    sampling_set = (
        getattr(gc, "top_p", None) not in (None, 1.0)
        or getattr(gc, "top_k", None) not in (None, 0, 50)
        or getattr(gc, "temperature", None) not in (None, 1.0)
    )
    if sampling_set and not getattr(gc, "do_sample", False):
        gc.do_sample = True


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #

# Files copied verbatim from source model dir — preserves preprocessing config
# untouched by calibration. Matches NVIDIA's copy list.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
)

# Tokenizer files copied verbatim from source. Quantization only rewrites model
# weights (+ config.json / generation_config.json); it never alters the
# tokenizer, so the source tokenizer files are byte-for-byte authoritative.
# Copying them verbatim — instead of trusting tokenizer.save_pretrained() to
# round-trip cleanly — makes it impossible for calibration-time truncation state
# (max_length=512, truncation_strategy, ...) to leak into the deployed
# tokenizer_config.json. That leak silently caps inference sequence length
# (e.g. MME benchmark truncated to 512). Non-existent files are skipped, so
# listing both fast (tokenizer.json) and slow (vocab.json/merges.txt) artifacts
# is safe across tokenizer variants.
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_quantized_model(
    model,
    tokenizer,
    processor,
    model_dir: str,
    output_dir: str,
) -> None:
    """Save a quantized model with HF-compatible layout.

    Mirrors NVIDIA's export tail:
      1. Apply transformers >= 5.x WARs.
      2. ``export_hf_checkpoint`` writes weights + ``config.json`` +
         ``generation_config.json`` + ``hf_quant_config.json``.
      3. Save tokenizer + processor (so the full file set always exists).
      4. Overwrite the tokenizer and preprocessor config files with verbatim
         source copies. This is the authoritative step: quantization never
         changes these, so the source files are correct by definition, and a
         verbatim copy guarantees calibration's ``max_length=512`` truncation
         state can never leak into ``tokenizer_config.json``. Only allow-listed
         filenames are overwritten — the quantized weights, ``config.json``,
         ``generation_config.json`` and ``hf_quant_config.json`` from step 2 are
         left untouched.
    """
    from modelopt.torch.export import export_hf_checkpoint

    fix_generation_config_for_strict_validate(model)
    normalize_tied_weights_keys(model)

    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=output_dir)

    tokenizer.save_pretrained(output_dir)

    if processor is not None:
        processor.save_pretrained(output_dir)

    # Overwrite tokenizer + processor configs with verbatim source copies so
    # quantization/calibration cannot alter them. This is the authoritative
    # final write: it runs after both save_pretrained() calls and only touches
    # the allow-listed filenames, so weights, config.json, generation_config.json
    # and hf_quant_config.json produced by export_hf_checkpoint are untouched.
    for fname in (*TOKENIZER_FILES, *PROCESSOR_FILES):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
