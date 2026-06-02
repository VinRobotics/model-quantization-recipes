# qwen36-quantize

Multi-scheme quantization pipeline for **Qwen3.6-27B** using
[llm-compressor](https://github.com/vllm-project/llm-compressor).

Supports **FP8 · NVFP4 · INT8 · INT4** with no hardcoded scheme logic — all
scheme definitions live in `configs/schemes.yaml`.

Tested on **2× NVIDIA RTX 5090** (Blackwell, 32 GiB each).

---

## Repository structure

```
qwen36-quantize/
├── quantize.py              # Main quantization script
├── quantize.sh              # Shell wrapper (executable)
├── configs/
│   └── schemes.yaml         # Scheme definitions: FP8, NVFP4, INT8, INT4
├── requirements.txt         # Python dependencies (pinned)
└── requirements-torch.txt   # PyTorch install commands (pick your CUDA version)
```

---

## Tested environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| GPU | 2× NVIDIA RTX 5090 (32 GiB, Blackwell SM 120) |
| CUDA | 12.8 (nvcc 12.8.61) |
| Python | 3.11 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.6.0 |
| llm-compressor / compressed-tensors | 0.10.0.1 / 0.14.0.1 |
| vLLM (inference) | 0.19.1 |

---

## Installation

**1. Create and activate a virtual environment**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

**2. Install PyTorch**

See `requirements-torch.txt` for the exact `pip install` command matching your
CUDA version. RTX 5090 (Blackwell SM 120) requires CUDA 12.4 or 12.8.

**3. Install project dependencies**

```bash
pip install -r requirements.txt
```

> NVFP4 scheme additionally requires vLLM ≥ 0.19 for inference.

---

## Compatibility patches

Some versions of llm-compressor (0.10.x) have minor incompatibilities with
transformers ≥ 5.6. The `quantize.sh` script detects and patches them
automatically at runtime:

- **Patch 1** — `TORCH_INIT_FUNCTIONS` removed from `transformers.modeling_utils`
- **Patch 2** — `_get_no_split_modules("auto")` replaced by `_no_split_modules`

Set `--venv_path` to your virtual environment directory so the script can
locate the installed packages.

---

## Usage

### Shell wrapper (recommended)

```bash
# FP8 quantization (default)
./quantize.sh

# NVFP4 — Blackwell only (RTX 5090)
./quantize.sh --scheme nvfp4 --model_path ./Qwen3.6-27B

# INT8
./quantize.sh --scheme int8 --model_path ./Qwen3.6-27B --output_path ./Qwen3.6-27B-INT8

# INT4 (weight-only W4A16)
./quantize.sh --scheme int4
```

Full flag reference:

```
--scheme              fp8 | nvfp4 | int8 | int4   (default: fp8)
--model_path          Path to source model         (default: ./Qwen3.6-27B)
--output_path         Destination directory        (default: <model>-<SCHEME>)
--schemes_config      Path to schemes YAML         (default: configs/schemes.yaml)
--venv_path           Venv path for patches        (default: ./venv)
--num_calib_samples   Calibration samples          (default: 512)
--max_seq_len         Max token length             (default: 1024)
--max_memory_per_gpu  VRAM cap per GPU in GiB      (default: 28)
```

### Direct Python

```bash
python quantize.py \
    --model_path  ./Qwen3.6-27B \
    --scheme      fp8 \
    --output_path ./Qwen3.6-27B-FP8
```

---

## Adding or modifying schemes

All scheme logic is in `configs/schemes.yaml`. To add a custom scheme, append
an entry following the existing structure:

```yaml
my_scheme:
  scheme: FP8          # llm-compressor scheme string
  targets: Linear
  ignore:
    - lm_head
    - re:model\.visual\..*
```

Then run:

```bash
./quantize.sh --scheme my_scheme
```

No Python edits required.

---

## Scheme comparison

| Scheme | llm-compressor string | Weight bits | Activation bits | Hardware requirement |
|--------|-----------------------|-------------|-----------------|----------------------|
| `fp8`  | `FP8` | 8 | 8 (static) | Ampere SM 89+ |
| `nvfp4` | `NVFP4` | 4 | 4 (dynamic) | **Blackwell SM 120+** |
| `int8` | `W8A8` | 8 | 8 | Turing SM 75+ |
| `int4` | `W4A16` | 4 | 16 (BF16) | Any GPU |

---

## Architecture note — Qwen3.6-27B

The model interleaves **3 × DeltaNet (linear attention)** + **1 × full GQA**
every 4 layers (16 groups × 4 = 64 layers total). The DeltaNet recurrent cores
are numerically sensitive and are excluded from quantization in every scheme.
The vision encoder is also preserved in BF16 across all schemes.

---

## Authors

- **VinRobotics AI Team**
- HuggingFace: [vrfai](https://huggingface.co/vrfai)

## Published weights

The quantized checkpoints are published and available on Hugging Face:

| Checkpoint | Format |
|------------|--------|
| [`vrfai/Qwen3.6-27B-NVFP4`](https://huggingface.co/vrfai/Qwen3.6-27B-NVFP4) | NVFP4 |
| [`vrfai/Qwen3.6-27B-FP8`](https://huggingface.co/vrfai/Qwen3.6-27B-FP8) | FP8 |

## Credits

- Original model: [Qwen Team](https://huggingface.co/Qwen) (Alibaba Group)
- Quantization framework: [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
