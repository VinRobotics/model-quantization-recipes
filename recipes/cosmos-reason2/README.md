# cosmos-reason2

Cosmos Reason2 quantization recipe using llmcompressor and Hugging Face export flow.

## Status

- Owner: unassigned
- Model family: `Cosmos-Reason2` (2B, 8B)
- Quantization presets: `nvfp4`
- Runtime target: exported Hugging Face checkpoint

## Published Weights

| Checkpoint | Variant | Format |
|------------|---------|--------|
| [`vrfai/cosmos-reason2-2b-nvfp4`](https://huggingface.co/vrfai/Cosmos-Reason2-2B-NVFP4) | Cosmos Reason2 2B | NVFP4 |
| [`vrfai/cosmos-reason2-8b-nvfp4`](https://huggingface.co/vrfai/Cosmos-Reason2-8B-NVFP4) | Cosmos Reason2 8B | NVFP4 |

## Files

    .
    ├── quantize_cosmos_reason2.py  # Main quantization script
    ├── quantize.sh                 # Runner script
    ├── requirements.txt            # Python dependencies (excluding PyTorch)
    ├── requirements-torch.txt      # PyTorch installation guide
    └── README.md

## Setup

**Step 1 — Install PyTorch** for your CUDA version (see requirements-torch.txt):

| CUDA | Command |
|------|---------|
| 12.1 | pip install torch==2.10.0+cu121 torchvision==0.25.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 |
| 12.4 / 12.8 / 13.x | pip install torch==2.10.0+cu124 torchvision==0.25.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 |

**Step 2 — Install remaining dependencies:**

    pip install -r requirements.txt

## Quick Start

    # Quantize 2B model
    MODEL_PATH=/path/to/Cosmos-Reason2-2B OUTPUT_PATH=/path/to/output ./quantize.sh

    # Quantize 8B model
    MODEL_PATH=/path/to/Cosmos-Reason2-8B OUTPUT_PATH=/path/to/output ./quantize.sh

## CLI Reference

    quantize_cosmos_reason2.py [options]

      --model_path PATH           Local path to the model checkpoint (required)
      --output_path PATH          Destination for quantized model (required)
      --num_calib_samples N       Number of calibration samples (default: 512)
      --max_seq_len N             Maximum sequence length (default: 1024)
      --dataset_id ID             HuggingFace dataset (default: abisee/cnn_dailymail)
      --dataset_config CFG        Dataset config (default: 3.0.0)
      --dataset_split SPLIT       Dataset split (default: train)
      --device STRATEGY           device_map strategy (default: auto)
      --max_memory_per_gpu GiB    VRAM cap per GPU in GiB (default: 30)

## Notes

- Applies to both Cosmos Reason2 2B and 8B variants.
- Quantizes LLM backbone only (Linear layers); visual encoder kept in bf16.
- Calibration uses text-only data since visual encoder is excluded.
- Scheme fixed to NVFP4 — requires Blackwell GPU (compute capability >= 10.0).

## Tested Environments

- **OS:** Ubuntu 22.04 LTS
- **Hardware:** 2x NVIDIA GeForce RTX 5090 32GB
- **Python:** 3.10.19
- **PyTorch:** 2.10.0+cu128
- **CUDA:** 12.8 (Driver 590.48.01)
