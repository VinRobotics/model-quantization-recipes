# internvla-n1-dualvln

FP8 / NVFP4 quantization and TensorRT-Edge-LLM deployment for InternVLA-N1-DualVLN, a
dual-system vision-language navigation model, on NVIDIA Jetson Thor.

## Status

Native support landed in TensorRT-Edge-LLM:
[NVIDIA/TensorRT-Edge-LLM#193](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/193)
(pending review, tracked by
[NVIDIA/TensorRT-Edge-LLM#190](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/190)). This
recipe used to repackage the checkpoint and compute the `z_latents` bridge on the host in
Python; both steps are gone now that TensorRT-Edge-LLM exports the checkpoint directly and
folds the bridge (`final_norm` + `cond_projector`) into the graph. What is left here is the one
thing that stays outside TensorRT-Edge-LLM: building a navigation-domain calibration set.

**Measured on Jetson Thor, 199 R2R val_unseen episodes, closed-loop:**

| | prefill | decode | control rate | engine | SR vs PyTorch (69.8%) |
|---|---|---|---|---|---|
| TensorRT FP8 | 90.6 ms | 32.8 ms | 61.3 ms (16.3 Hz) | 7.10 GB | 68.3% (p = 0.728) |
| TensorRT NVFP4 | 75.4 ms | 20.3 ms | 55.4 ms (18.0 Hz) | 4.45 GB | 67.8% (p = 0.572) |

Neither differs from PyTorch significantly. **This replaces the recipe's earlier
recommendation.** The old version gated acceptance on `z_latents` cosine and, on that basis,
recommended FP8 and ruled NVFP4 out (cosine 0.647 against a 0.99 gate). Closed-loop SR shows
that gate does not predict the outcome it stands in for — full validation and the "no offline
metric predicts SR" finding are in the PR. Pick NVFP4 for speed and size, FP8 for a wider
margin; both are viable.

## What this recipe does

Build a calibration set of realistic navigation prompts. Calibrating the quantized backbone on
its own prompt domain, rather than generic news text, measurably changes activation scales —
the same FP8 recipe moved from trajectory cosine 0.909 (`cnn_dailymail`, the CLI's default) to
0.978 (navigation prompts) in earlier testing on this model.

```bash
huggingface-cli download InternRobotics/InternData-N1 \
    vln_ce/raw_data/r2r/train/train.json.gz --repo-type dataset \
    --local-dir $CALIB_DATA_ROOT
# gated on Hugging Face -- accept the dataset terms and `huggingface-cli login` first

python quantize/build_calib_jsonl.py \
    --train_json $CALIB_DATA_ROOT/vln_ce/raw_data/r2r/train/train.json.gz \
    --output $CALIB_DATA_ROOT/nav_calib.jsonl
```

Everything after that is the standard TensorRT-Edge-LLM flow, in its own environment:

```bash
# Point the CLI's default text_dataset at the local file instead of the Hub.
export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=$CALIB_DATA_ROOT/nav_calib.jsonl

tensorrt-edgellm-quantize llm --model_dir $INTERNVLA_CKPT \
    --output_dir $QUANT_CKPT --quantization {fp8,nvfp4}
tensorrt-edgellm-export $QUANT_CKPT $ONNX_DIR

export EDGELLM_PLUGIN_PATH=.../libNvInfer_edgellm_plugin.so
export __LUNOWUD="-cask_fusion:max_num_epilogues=1"   # NVFP4 at maxBatchSize 1 only
llm_build --onnxDir $ONNX_DIR/llm --engineDir $ENGINE_DIR/llm \
    --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096
```

System 1 (the trajectory expert) is not part of this checkpoint's quantization — it stays
BF16, built separately with `trtexec` — and the async runtime
(`internvla_n1_dual_system_inference` / `internvla_n1_dual_system_server`) that drives both
systems is not part of this repo either; it ships with TensorRT-Edge-LLM. See
[`experimental_models/internvla_n1/README.md`](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/experimental_models/internvla_n1/README.md)
in that repo (or the PR branch, until it merges) for the full export → build → run flow and the
resident-server protocol for driving it from a Python agent.

## Why calibration text needs no special tokens here

An earlier version of this pipeline appended four trajectory-query placeholder tokens to every
calibration prompt, because its own driver ran the System-2 → System-1 bridge forward pass
during calibration. `tensorrt-edgellm-quantize` does not: it loads System 2 as a stock
Qwen2.5-VL (see `internvla_n1_loader.py` in TensorRT-Edge-LLM) and calibrates with ordinary
text forward passes, never touching the bridge. Appending tokens the tokenizer does not even
have registered yet — they are added later, at export time — would only add noise.

## What is and is not quantized

Quantized: the System 2 LLM backbone. Never quantized: System 1 (`traj_dit`, memory block),
and the bridge (`cond_projector`, `latent_queries`) — four rows through a
Linear/GELU/Linear, kept at source precision because quantizing them saves nothing measurable
and puts error directly on the tensor System 1 steers by.

**NVFP4 with the vision tower is blocked, permanently.** The Qwen2.5-VL ViT MLP has
`intermediate_size = 3420`, and 3420 / 16 = 213.75 — not divisible by the NVFP4 block size.
Only the LLM backbone is quantized here, so this does not apply, but it is worth knowing if you
extend this to a strategy that includes the vision tower.

## Tested environment

Jetson Thor, JetPack 7.1 (TensorRT 10.13.3.9, CUDA 13). `pip install -e ".[tools]"` in the
TensorRT-Edge-LLM checkout pulls `nvidia-modelopt` and `datasets`; without it
`tensorrt-edgellm-quantize` fails at import.
