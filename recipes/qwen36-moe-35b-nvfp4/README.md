# qwen36-moe-35b-nvfp4
NVFP4 quantization for Qwen3.6-35B-A3B (hybrid linear-attention / full-attention MoE) via nvidia-modelopt.

## Status
- Owner: unassigned
- Model family: `Qwen3.6-35B-A3B`
- Quantization preset: `nvfp4`
- Runtime target: SGLang with `modelopt_fp4` backend

## Published weights
| Checkpoint | Variant | Format |
|------------|---------|--------|
| [`vrfai/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/vrfai/Qwen3.6-35B-A3B-NVFP4) | Qwen3.6 35B-A3B | NVFP4 |

## Setup

### Option A — uv (recommended)
```bash
uv sync --extra qwen36-moe-35b-nvfp4
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

Edit `MODEL_PATH` and `QUANT_DTYPE` at the top of `quantize.sh`, then run:

```bash
./quantize.sh
```

Or call Python directly:

```bash
python3 quantize.py \
    --model_path  ./model \
    --output_path ./model-nvfp4 \
    --quant_dtype nvfp4
```

## CLI Reference

```
quantize.py [options]

  --model_path PATH           Local model directory (required)
  --output_path PATH          Destination for quantized checkpoint (required)
  --quant_dtype DTYPE         int8 | fp8 | nvfp4  (default: int8)
  --num_calib_samples N       Calibration samples (default: 256)
  --max_seq_len N             Max token length per sample (default: 1024)
  --device STRATEGY           device_map strategy (default: auto)
```

## Serving with SGLang

```bash
sglang serve \
  --model-path vrfai/Qwen3.6-35B-A3B-NVFP4 \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 1 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --quantization modelopt_fp4
```

## Exclusion Logic

Exclusions are derived at runtime from `model.named_modules()` and adapt to the loaded architecture without hardcoding layer indices:

- `lm_head`, `embed_tokens`, all vision modules
- All normalization layers (`norm` in name)
- MoE gate / router / shared expert gate layers
- `linear_attn` SSM components: `conv1d`, `A_log`, `dt_bias`, `in_proj` — stateful, not quantizable

## Output Files

```
<output_path>/
├── config.json            # Model config with quantization metadata
├── hf_quant_config.json   # ModelOpt quantization metadata
├── *.safetensors          # Quantized weights
├── tokenizer / processor  # Copied verbatim from the base checkpoint
├── generation_config.json # Copied verbatim from the base checkpoint
└── quant_log.txt          # Per-layer quantization summary
```

Everything except `config.json`, `hf_quant_config.json` and the weight shards is
copied from the base checkpoint rather than re-serialized.
`tokenizer.save_pretrained()` rebuilds the config from the live Python object and
silently drops whatever the loaded class does not model, which leaves the
quantized checkpoint preprocessing inputs differently from the model it was
derived from. Copying the originals is lossless and version-independent.

## Memory Notes

- Model is loaded in `bfloat16` before quantization.
- `torch.cuda.empty_cache()` is called between calibration samples to reduce OOM risk.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by `quantize.sh` to reduce fragmentation.
- If OOM during calibration: lower `--num_calib_samples` or `--max_seq_len`.

## Files

```
.
├── quantize.py            # Main quantization script
├── quantize.sh            # Runner shell script
├── requirements.txt       # Python dependencies (excluding torch)
├── requirements-torch.txt # PyTorch install instructions by CUDA version
└── README.md
```

Dependencies are also defined in the repository root `pyproject.toml` via the `qwen36-moe-35b-nvfp4` extra.

## Tested Environments

- **OS:** Ubuntu 22.04 LTS
- **Hardware:** 1x NVIDIA H100 80GB HBM3
- **PyTorch:** 2.5.1+cu121 (CUDA 12.1) or 2.5.1+cu124 (CUDA 12.4)
- **nvidia-modelopt:** 0.39.0
- **transformers:** 5.5.4
