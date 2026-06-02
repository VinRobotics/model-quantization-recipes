# QAD — Quantization-Aware Distillation for ASR

Train an INT4-quantized ASR student model via knowledge distillation from a full-precision teacher.

Default target: **Qwen3-ASR** family, but the codebase is model-agnostic — see [Adapting to Other Models](#adapting-to-other-models).

---

## Requirements

```bash
pip install torch torchaudio soundfile numpy pyyaml jiwer
pip install nvidia-modelopt        # INT4 student loading
pip install safetensors transformers
```

---

## Quick Start

### 1. Copy and edit the config

```bash
cp configs/qad_example.yaml configs/my_run.yaml
# Fill in: paths.teacher_ckpt, student_ckpt, data_dir, output_dir, eval_vivos, eval_libri
```

### 2. Run the full pipeline

```bash
# Auto-detect GPU count
bash run_qad.sh --config configs/my_run.yaml

# Explicit GPU count
bash run_qad.sh --config configs/my_run.yaml --nproc 8
```

### 3. Run stages individually

```bash
# Stage 1: Generate pseudo-labels with teacher (single GPU)
python qad_train.py --config configs/my_run.yaml --mode prepare

# Stage 2: Sanity check — verify loss + backward pass (single GPU)
python qad_train.py --config configs/my_run.yaml --mode sanity

# Stage 3: Full QAD training (multi-GPU DDP)
torchrun --nproc_per_node=4 qad_train.py --config configs/my_run.yaml --mode train
```

---

## Pipeline Overview

```
Stage 1 (prepare)
  Teacher.transcribe(wav) → pseudo_labels.jsonl

Stage 2 (sanity)
  Overfit 4 samples for 5 steps → verify loss is finite

Stage 3 (train)
  For each batch:
    logits_T = Teacher.forward(input)   # frozen
    logits_S = Student.forward(input)   # INT4 fake-quant, decoder trainable
    loss = α·KL(p_T‖p_S) + (1-α)·CE(logits_S, pseudo_labels)
    loss.backward()
  WER eval on VIVOS + LibriSpeech every save_interval steps
```

---

## Eval Data Layout

```
eval_vivos/
    prompts.txt         ← lines: "UTT_ID transcript text"
    waves/
        UTT_ID.wav      ← flat or nested subdirectories

eval_libri/             ← same layout
    prompts.txt
    waves/
```

---

## Outputs

```
output_dir/
    step_0001000/           ← checkpoint every save_interval steps
        model.safetensors
        config.json
        tokenizer files...
        trainer_state.pt    ← optimizer + scheduler state for resume
    step_0020000_final/     ← final checkpoint
    wer_log.jsonl           ← one JSON line per eval
    wer_summary.txt         ← formatted table with ★ best steps
    tensorboard/            ← TensorBoard logs
    logs/                   ← shell script run logs
```

Training automatically **resumes** from the latest checkpoint if `output_dir` already contains checkpoints.

---

## Key Config Options

| Section | Key | Description |
|---------|-----|-------------|
| `paths` | `teacher_ckpt` | Full-precision teacher (any size) |
| `paths` | `student_ckpt` | PTQ quantized student (INT4, INT8, FP8, …) |
| `paths` | `custom_src` | Custom model source dir; `null` = standard HF |
| `model` | `model_class` | HuggingFace model class name |
| `model` | `quantization_backend` | `"modelopt"` for NVIDIA modelopt checkpoints; `null` for bitsandbytes / GPTQ / AWQ / HQQ / standard HF |
| `model` | `inner_module` | Sub-module wrapping the LM (`"thinker"` for Qwen3-ASR, `null` for standard HF) |
| `model` | `audio_end_token_id` | Token marking start of transcript; `null` = auto-detect |
| `model` | `audio_prompt_override` | Audio placeholder string; `null` = auto-detect from processor |
| `training` | `alpha_kd` | `1.0` = pure KD, `0.5` = KD + CE mix |
| `training` | `frozen_keywords` | List of substrings — matching params are frozen |

---

## Pretrained Checkpoints

QAD-enhanced checkpoints are publicly available on Hugging Face:

| Format | HuggingFace |
|--------|-------------|
| INT4 AWQ + QAD | [vrfai/Qwen3-ASR-0.6B-int4-QAD](https://huggingface.co/vrfai/Qwen3-ASR-0.6B-int4-QAD) |
| INT8 SmoothQuant + QAD | [vrfai/Qwen3-ASR-0.6B-int8-QAD](https://huggingface.co/vrfai/Qwen3-ASR-0.6B-int8-QAD) |

---

## Adapting to Other Models

To use a different model family, update the `model:` section in your config:

```yaml
paths:
  custom_src: null   # use standard HuggingFace transformers

model:
  model_class:      "WhisperForConditionalGeneration"
  model_module:     "transformers"
  processor_class:  "WhisperProcessor"
  processor_module: "transformers"
  eval_model_class: "YourEvalWrapper"   # must expose .transcribe()
  eval_model_module: "your_package"
  inner_module:     null                # no inner wrapper
  audio_end_token_id: 50257             # your model's decoder_start_token_id
  audio_prompt_override: ""             # adjust to your model's format
  quantization_backend: null    # null = standard HF (bitsandbytes, GPTQ, AWQ, HQQ…)
  processor_kwargs: {}
```

The only hard requirement is that `eval_model_class` exposes:
```python
results = model.transcribe(audio=(np_array, sample_rate), language=None)
text = results[0].text
```

---

## Pretrained Checkpoints

QAD-enhanced checkpoints are publicly available on Hugging Face:

| Format | HuggingFace |
|--------|-------------|
| INT4 AWQ + QAD | [vrfai/Qwen3-ASR-0.6B-int4-QAD](https://huggingface.co/vrfai/Qwen3-ASR-0.6B-int4-QAD) |
| INT8 SmoothQuant + QAD | [vrfai/Qwen3-ASR-0.6B-int8-QAD](https://huggingface.co/vrfai/Qwen3-ASR-0.6B-int8-QAD) |
