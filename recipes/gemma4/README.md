# gemma4
Generic Gemma4 quantization recipe using NVIDIA ModelOpt and Hugging Face export flow.
## Status
- Owner: unassigned
- Source branch: generalized from `origin/gemma4_31b` in `qwen-asr-optimization`
- Migration state: refactored for broader Gemma4 reuse
## Scope
- Model family: `Gemma4`
- Quantization presets: `fp8`, `nvfp4`, `mxfp8`, `int4_awq`, `int8_sq`
- Runtime target: exported Hugging Face checkpoint
## Published weights

Optimized Gemma 4 checkpoints are published in the [VRFAI Gemma 4 Optimized collection](https://huggingface.co/collections/vrfai/gemma-4-optimized).

| Checkpoint | Variant | Format |
|------------|---------|--------|
| [`vrfai/gemma-4-E4B-it-fp8`](https://huggingface.co/vrfai/gemma-4-E4B-it-fp8) | Gemma 4 E4B IT | FP8 |
| [`vrfai/gemma-4-31B-it-fp8`](https://huggingface.co/vrfai/gemma-4-31B-it-fp8) | Gemma 4 31B IT | FP8 |
## Files
## Setup

### Option A — uv (recommended)
```bash
uv sync --extra gemma4
```

### Option B — pip

**Step 1 — Install PyTorch** (pick the line matching your CUDA version):

```bash
# CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 / 12.x / 13.x
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch==2.5.1 torchvision==0.20.1
```

**Step 2 — Install remaining dependencies:**

```bash
pip install -r requirements.txt
```
## Quick Start
You can run the quantization pipeline by providing the necessary environment variables. Set `MODEL_PATH` to your downloaded Gemma4 checkpoint directory.
```bash
MODEL_PATH=/path/to/gemma4-checkpoint QUANTIZATION=fp8 ./quantize.sh
```
## Environment Variables
The `quantize.sh` script accepts the following environment variables to override defaults:
- `MODEL_PATH`: Local path to the model checkpoint.
- `QUANTIZATION`: Preset to apply (default: `fp8`).
- `MODEL_DTYPE`: Base model data type (default: `bfloat16`).
- `OUTPUT_PATH`: Directory for quantized output.
- `NUM_CALIB_SAMPLES`: Number of calibration samples (default: `512`).
- `MAX_SEQ_LEN`: Maximum sequence length for calibration (default: `1024`).
## CLI Examples
Run the Python script directly for finer control:
```bash
python quantize_gemma4.py \
  --model_path /path/to/gemma4-model \
  --output_path /path/to/output \
  --quantization nvfp4 \
  --model_dtype bfloat16
```
```bash
python quantize_gemma4.py \
  --model_path /path/to/gemma4-model \
  --output_path /path/to/output \
  --quantization int4_awq \
  --dataset_id abisee/cnn_dailymail \
  --dataset_text_column article
```
## Using published checkpoints
### Transformers
```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model_id = "vrfai/gemma-4-31B-it-fp8"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain FP8 quantization in two sentences."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```
### vLLM
```bash
vllm serve vrfai/gemma-4-31B-it-fp8 \
  --quantization modelopt \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --async-scheduling \
  --trust-remote-code
```
## Notes
- The script auto-detects whether the checkpoint should load through `AutoModelForCausalLM` or `AutoModelForImageTextToText`.
- Calibration defaults to `abisee/cnn_dailymail` and the `article` text column.
- Default exclusion rules are derived from the loaded model config and applied on top of the selected quantization preset.
- `quantize_decisions.txt` is written into the output folder for each run.
## Tested Environments
- **OS:** Ubuntu 22.04 LTS
- **Hardware:** 1x NVIDIA H100 80GB HBM3
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1) or 2.5.1+cu124 (CUDA 12.4)
- **nvidia-modelopt:** 0.39.0
