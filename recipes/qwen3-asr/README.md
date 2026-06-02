# qwen3-asr-optimization

Post-training quantization (PTQ) and inference optimization for
[Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
targeting two deployment backends:

| Backend | Hardware | Formats | Use case |
|---------|----------|---------|----------|
| **[vllm/](vllm/)** | NVIDIA RTX 5090 (x86) | FP8, NVFP4 | Online HTTP serving |
| **[trt-edgellm/](trt-edgellm/)** | Jetson Orin Nano 8 GB | INT8, INT4 | On-device edge inference |

---

## Model Architecture

![Model Architecture](assets/modelArchitectures.png)

Qwen3-ASR 1.7B is a multimodal ASR model. Three components live under `thinker`:

| Component | Description | Params | Quantized |
|-----------|-------------|--------|-----------|
| `audio_tower` | 24-layer audio encoder (CNN + attention + GELU projector) | ~0.3 B | No |
| `model` | 28-layer Qwen3-based LLM decoder (GQA 16/8 heads, SwiGLU) | ~1.4 B | **Yes** |
| `lm_head` | Output projection, tied weights with `embed_tokens` | shared | No |

Only the LLM (`model`) is quantized: it accounts for ~82% of parameters while
`audio_tower` handles sensitive acoustic features and `lm_head` shares weights
with `embed_tokens` via weight tying.

---

## Repository Structure

```
qwen-asr-optimization/
├── .gitignore
├── .gitmodules                  # Git submodule registry
├── Makefile                     # Top-level command panel
├── README.md                    ← you are here
├── requirements.txt
│
├── assets/
│   ├── modelArchitectures.png
│   └── workFlowTensorRTLLM.png
│
├── external/                    # Third-party dependencies
│   ├── init_submodules.sh       # 1-click: clone Qwen3-ASR + TRT-Edge-LLM
│   ├── qwen3_asr/               # [Submodule] github.com/QwenLM/Qwen3-ASR
│   └── tensorrt_edge_llm/       # [Submodule] TensorRT-Edge-LLM v0.6.0
│
├── vllm/                        # ── Path 1: Online serving (RTX 5090)
│   ├── README.md
│   ├── benchmark_vivos.py
│   ├── export.py
│   ├── quantize.py
│   └── scripts/
│       ├── 01_quantize.sh
│       ├── 02_export.sh
│       └── 03_serve.sh
│
├── trt-edgellm/                 # ── Path 2: Edge inference (Jetson Orin Nano)
│   ├── README.md
│   ├── asr_demo.html            # Browser-based demo UI for the ASR server
│   ├── asr_server.py            # FastAPI server exposing the TRT-LLM engine
│   ├── benchmark.py
│   ├── inference.py
│   ├── quantize.py
│   └── scripts/
│       ├── 01_quantize.sh
│       ├── 02_export_onnx.sh
│       ├── 03_build_engine.sh
│       └── 04_benchmark.sh
│
└── llm-qad/                     # ── Path 3: Quantization-Aware Distillation (QAD)
    ├── README.md
    ├── qad_train.py             # Main training entry-point
    ├── requirements.txt
    └── configs/
        ├── README.md
        ├── qad_example.yaml     # Example QAD configuration
        └── run_qad.sh           # Launch script for QAD training
```

---

## Results Summary

### vLLM — RTX 5090

| Model | WER | RTF | Throughput (conc=1) | Weights |
|-------|-----|-----|---------------------|---------|
| BF16  | 7.34% | 0.0190 | 15.42 req/s | 3.87 GB |
| FP8   | 7.60% | 0.0152 | 19.37 req/s | 2.55 GB |
| NVFP4 | 10.73% | 0.0186 | 15.77 req/s | 1.99 GB |

### TRT-Edge-LLM — Jetson Orin Nano 8 GB

Baseline: BF16 WER 7.34% (measured on x86 + RTX 5090)

| Format | WER | RTF | Throughput | RAM |
|--------|-----|-----|-----------|-----|
| INT8 SmoothQuant | 9.07% | 0.2190 | 1.29 samples/s | 4.2 GB |
| INT4 AWQ         | 8.69% | 0.1641 | 1.72 samples/s | 3.3 GB |

## Notes

* **Memory difference (Jetson vs RTX):**

  * Jetson uses **unified memory (CPU + GPU shared)** → includes *weights + activations + runtime buffers*
  * RTX reports **weights only (VRAM)** → not directly comparable

* **Edge trade-off:**

  * **INT4 AWQ** provides the best speed with a moderate WER increase (**~+1.3–1.7% vs BF16 baseline**)

---

## Quick Start

### 1. Install dependencies

```bash
cd qwen-asr-optimization

# Install Python requirements from the repository root
uv sync --extra qwen3-asr

# Clone Qwen3-ASR and TensorRT-Edge-LLM (into external/)
make setup-submodules
```

### 2. Download model weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
```

### 3. Choose your path

- **vLLM (RTX 5090 / Ada/Hopper)** → [vllm/README.md](vllm/README.md)
- **TRT-Edge-LLM (Jetson Orin Nano)** → [trt-edgellm/README.md](trt-edgellm/README.md)
- **LLM-QAD (quantization-aware distillation)** → [llm-qad/README.md](llm-qad/README.md)

Before running, update the paths and settings in the relevant config file (`llm-qad/configs/qad_example.yaml` for QAD, or the corresponding `.sh` script for TRT-Edge-LLM) to match your local environment — key fields to fill in are the model checkpoint path, dataset path, and output directory.

---

## Pretrained Checkpoints (vLLM)

| Format | HuggingFace |
|--------|-------------|
| FP8    | [vrfai/qwen3asr-fp8](https://huggingface.co/vrfai/qwen3asr-fp8) |
| NVFP4  | [vrfai/qwen3asr-nvfp4](https://huggingface.co/vrfai/qwen3asr-nvfp4) |
| INT8   | [vrfai/qwen3asr-int8](https://huggingface.co/vrfai/qwen3asr-int8) |
| INT4   | [vrfai/qwen3asr-int4](https://huggingface.co/vrfai/qwen3asr-int4) |

---

## References

- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
- [Qwen3-ASR Model](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [NVIDIA ModelOpt Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)
- [TensorRT-Edge-LLM v0.6.0](https://nvidia.github.io/TensorRT-Edge-LLM/0.6.0/developer_guide/getting-started/installation.html)
- [vLLM Documentation](https://docs.vllm.ai)
