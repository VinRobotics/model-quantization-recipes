# vllm — Online Serving (FP8 / NVFP4)

Post-training quantization of Qwen3-ASR 1.7B to FP8 and NVFP4
using [NVIDIA ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/),
served via [vLLM](https://docs.vllm.ai).

---

## Prerequisites

**Hardware:** NVIDIA RTX 5090 (Ada/Hopper for FP8; Blackwell for NVFP4)

```bash
# From repo root
uv sync --extra qwen3-asr
make -C recipes/qwen3-asr setup-submodules   # clones external/qwen3_asr and external/tensorrt_edge_llm

pip install vllm     # version must match your CUDA — see https://docs.vllm.ai
```

**Download model weights:**
```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
```

---

## Pipeline

```
scripts/01_quantize.sh  →  scripts/02_export.sh  →  scripts/03_serve.sh  →  benchmark_vivos.py
       (.pt)                   (vllm/ dir)             (HTTP server)           (WER/RTF)
```

> **Note:** NVIDIA ModelOpt performs *fake quantization* — it inserts
> Quantize/Dequantize nodes but computes in BF16. WER metrics from the
> quantized `.pt` checkpoint are accurate; latency/RTF figures only reflect
> real speedup after export and vLLM serving.

---

## Step 1 — Quantize

```bash
# FP8 (Max calibration — recommended)
bash scripts/01_quantize.sh \
    --model_path    /path/to/Qwen3-ASR-1.7B \
    --qwen_asr_root ../external/qwen3_asr \
    --format        fp8 \
    --algorithm     max \
    --output_pt     ./qwen3asr_thinker_fp8.pt

# NVFP4 (Max calibration)
bash scripts/01_quantize.sh \
    --model_path    /path/to/Qwen3-ASR-1.7B \
    --qwen_asr_root ../external/qwen3_asr \
    --format        nvfp4 \
    --algorithm     max \
    --output_pt     ./qwen3asr_thinker_nvfp4.pt
```

Or from repo root via Makefile:
```bash
make vllm-quantize-fp8    MODEL_PATH=/path/to/Qwen3-ASR-1.7B
make vllm-quantize-nvfp4  MODEL_PATH=/path/to/Qwen3-ASR-1.7B
```

**Calibration algorithm reference:**

| Algorithm | `--algorithm` | Notes |
|-----------|--------------|-------|
| Max (default) | `max` | Per-tensor symmetric, MaxCalibrator |
| MSE | `mse` | Minimises mean squared error of activations |
| AWQ Full | `awq_full` | Activation-aware weight quantization |

---

## Step 2 — Export

```bash
# FP8
bash scripts/02_export.sh \
    --model_path    /path/to/Qwen3-ASR-1.7B \
    --qwen_asr_root ../external/qwen3_asr \
    --quant_pt      ./qwen3asr_thinker_fp8.pt \
    --tmp_dir       ./qwen3asr_fp8_hf \
    --export_dir    ./qwen3asr_fp8_vllm

# NVFP4
bash scripts/02_export.sh \
    --model_path    /path/to/Qwen3-ASR-1.7B \
    --qwen_asr_root ../external/qwen3_asr \
    --quant_pt      ./qwen3asr_thinker_nvfp4.pt \
    --tmp_dir       ./qwen3asr_nvfp4_hf \
    --export_dir    ./qwen3asr_nvfp4_vllm
```

---

## Step 3 — Serve

`qwen-asr-serve` (provided by Qwen3-ASR) registers
`Qwen3ASRForConditionalGeneration` in the vLLM `ModelRegistry` before startup.

```bash
# Usage: bash scripts/03_serve.sh <model_dir> <format> [gpu_id] [gpu_memory_utilization]
bash scripts/03_serve.sh ./qwen3asr_fp8_vllm   fp8   0 0.7
bash scripts/03_serve.sh ./qwen3asr_nvfp4_vllm nvfp4 0 0.7
```

---

## Step 4 — Benchmark

```bash
python benchmark_vivos.py \
    --url   http://localhost:8000/v1/audio/transcriptions \
    --model ./qwen3asr_fp8_vllm
```

Optional: `--dataset <hf_dataset_id>`, `--split <split_name>`.

---

## Results

### Memory usage (`gpu_memory_utilization=0.7`, RTX 5090)

| Model | Weights | KV Cache |
|-------|---------|----------|
| BF16  | 3.87 GB | 16.58 GB |
| FP8   | 2.55 GB | 17.86 GB |
| NVFP4 | 1.99 GB | 18.43 GB |

### WER by calibration algorithm (760 VIVOS test samples)

| Algorithm | FP8 WER | NVFP4 WER |
|-----------|---------|-----------|
| BF16 baseline (7.34%) | — | — |
| **Max** (default) | **7.60%** | 10.73% |
| MSE | 7.67% | 9.75% |
| AWQ Full | 7.65% | **9.45%** |

### Throughput across concurrency levels (7168 VIVOS samples, Max)

| Concurrency | BF16 (req/s) | FP8 (req/s) | NVFP4 (req/s) |
|-------------|-------------|------------|--------------|
| 1   | 15.42 | 19.37 | 15.77 |
| 256 | 227.88 | 246.47 | 260.87 |
| 512 | 227.50 | 251.37 | 262.87 |
| 1024 | 219.63 | 248.24 | 268.25 |

> FP8 consistently outperforms BF16 at single-request latency. NVFP4 achieves
> the highest throughput at high concurrency due to its smaller memory footprint
> freeing more KV cache.

### Single-request latency (concurrency=1)

| Model | WER | RTF | Avg Latency | Throughput |
|-------|-----|-----|-------------|------------|
| BF16  | 7.34% | 0.0190 | 0.06 s | 15.42 req/s |
| FP8   | 7.60% | 0.0152 | 0.05 s | 19.37 req/s |
| NVFP4 | 10.73% | 0.0186 | 0.06 s | 15.77 req/s |

### LLM output cosine similarity vs BF16 (50 samples)

| Model | Cosine Similarity |
|-------|------------------|
| FP8   | 0.994 |
| NVFP4 | 0.945 |

---

## Pretrained Checkpoints

| Format | HuggingFace |
|--------|-------------|
| FP8    | [vrfai/qwen3asr-fp8](https://huggingface.co/vrfai/qwen3asr-fp8) |
| NVFP4  | [vrfai/qwen3asr-nvfp4](https://huggingface.co/vrfai/qwen3asr-nvfp4) |
