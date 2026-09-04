# qwen38-27b

Post-training quantization for **Qwen3.8-27B** (a `qwen3_5` dense vision-language
model) with [llm-compressor](https://github.com/vllm-project/llm-compressor),
calibrated on CNN/DailyMail and exported as `compressed-tensors` for vLLM.

Produces **FP8 · FP8-dynamic · NVFP4** checkpoints. The three published builds are
on the Hugging Face Hub — see [Published checkpoints](#published-checkpoints).

## Status

- Owner: VinRobotics AI Team
- Migration state: complete
- Scope of this recipe: quantization and checkpoint validation. Accuracy
  benchmarking is run outside this recipe; the measured results are quoted below
  for reference only.

## Scope

- Model: `Qwen/Qwen3.8-27B` (`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`)
- Quantization method: llm-compressor one-shot PTQ, SmoothQuant + `QuantizationModifier`
- Runtime target: vLLM, via the `compressed-tensors` checkpoint format
- Hardware tested: 1× NVIDIA H100 80 GB (SM 90)

## Repository structure

```
qwen38-27b/
├── quantize.py           # the pipeline
├── inspect_model.py      # meta-device architecture census; owns the _POLICY table
├── verify_ignore.py      # audits the resolved ignore list against a reference config
├── sanity_gen.py         # four greedy generations — catches a broken checkpoint fast
├── run_quantize.sh       # runner for one strategy (inline or SLURM)
├── run_all.sh            # sequential driver for every strategy
├── reference/
│   └── qwen35_official_fp8_config.json
├── requirements.txt
└── requirements-torch.txt
```

`inspect_model.py` owns the `_POLICY` table and `quantize.py` imports it, so the
audit and the run can never disagree about what is quantized.

## Tested environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| GPU | 1× NVIDIA H100 80 GB HBM3 (SM 90) |
| CUDA | 12.8 |
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.14.1 |
| llm-compressor / compressed-tensors | 0.13.0 / 0.18.0 |
| Datasets | 5.0.1 |
| vLLM (validation only) | 0.27.1 |

`transformers >= 5.8` is a hard floor. Anything older fails at load with
`checkpoint has model type 'qwen3_5' but Transformers does not recognize this
architecture` — the architecture simply does not exist in earlier releases.

## Installation

```bash
uv sync --extra qwen38-27b
```

Or with `pip`, in a fresh Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# 1. PyTorch first — pick the line matching your CUDA toolkit
#    (see requirements-torch.txt)
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt
```

`sanity_gen.py` needs vLLM, which ships its own PyTorch build. Install it into a
**separate** environment so it does not overwrite the one used to quantize.

## Environment variables

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `MODEL_PATH` | yes | — | Base Qwen3.8-27B checkpoint |
| `OUT_ROOT` | no | `./outputs` | Where checkpoints are written |
| `STRATEGY` | no | `fp8` | One strategy for `run_quantize.sh` |
| `STRATEGIES` | no | `fp8 nvfp4 fp8-dynamic` | Strategy list for `run_all.sh` |
| `OUTPUT_DIR` | no | `$OUT_ROOT/qwen38-27b-$STRATEGY` | Explicit destination |
| `QUANT_VENV` | no | — | Virtualenv to activate before running |
| `NUM_CALIB_SAMPLES` | no | `512` | Calibration samples |
| `MAX_SEQ_LEN` | no | `2048` | Calibration token budget |
| `MAX_MEMORY_PER_GPU` | no | `74` | Per-GPU VRAM cap, GiB |
| `CPU_OFFLOAD_GB` | no | `160` | Host RAM offered to accelerate as overflow |

## Quick start

```bash
export MODEL_PATH=/path/to/Qwen3.8-27B

# 1. Architecture audit — fast, no GPU, no weight load
python inspect_model.py --model_path "$MODEL_PATH" --dump-ignore ignore_list.json

# 2. Quantize one strategy
STRATEGY=fp8 bash run_quantize.sh       # or: STRATEGY=fp8 sbatch run_quantize.sh

# 3. ... or every published strategy, sequentially
bash run_all.sh
```

Direct Python, without the wrappers:

```bash
python quantize.py \
    --model_path         /path/to/Qwen3.8-27B      \
    --output_path        ./outputs/qwen38-27b-fp8  \
    --strategy           fp8                       \
    --num_calib_samples  512                       \
    --max_seq_len        2048                      \
    --max_memory_per_gpu 74                        \
    --cpu_offload_gb     160
```

`python quantize.py --help` lists every flag.

## Architecture, and what that implies

`inspect_model.py` instantiates the model on the **meta device** — the full
module tree, without reading 54 GiB of bf16 weights — and prints `print(model)`,
a Linear-layer census, and the resolved ignore list.

```
architectures : ['Qwen3_5ForConditionalGeneration']
text          : 64 layers, hidden 5120, intermediate 17408, vocab 248320
layer_types   : {'linear_attention': 48, 'full_attention': 16}
vision        : depth 27, hidden 1152
```

The text stack is **hybrid**: 16 × (3 × Gated DeltaNet → FFN, 1 × Gated Attention
→ FFN). That is what drives every quantization decision below.

| role | #mods | params | % Linear | action |
|------|------:|-------:|---------:|--------|
| `mlp_up` (gate/up) | 128 | 11.41 B | 43.75 % | **quantize** |
| `mlp_down` | 64 | 5.70 B | 21.87 % | **quantize** |
| `ssm_in` (`in_proj_qkv`, `in_proj_z`) | 96 | 4.03 B | 15.44 % | **quantize** |
| `ssm_out` | 48 | 1.51 B | 5.79 % | **quantize** |
| `attn_qkv` | 48 | 1.17 B | 4.50 % | **quantize** |
| `attn_out` | 16 | 0.50 B | 1.93 % | **quantize** |
| `lm_head` | 1 | 1.27 B | 4.88 % | keep bf16 |
| `vision` | 110 | 0.46 B | 1.75 % | keep bf16 |
| `ssm_decay` (`in_proj_a`, `in_proj_b`) | 96 | 0.02 B | 0.09 % | keep bf16 |
| **total** | **607** | **26.08 B** | | **93.28 % quantized** |

### The three things held at bf16, and why

- **`lm_head`** — a 5120 × 248320 projection feeding the softmax directly.
  Standard to exclude; it is the single most accuracy-sensitive layer.
- **Vision tower** — 456 M of 26 B (1.75 %). Quantizing it saves almost nothing,
  and every downstream answer is conditioned on its tokens.
- **`in_proj_a` / `in_proj_b`** — the interesting one. These are 5120 → 48
  projections inside each Gated DeltaNet, producing the per-head decay `a` and
  the delta-rule `beta` that drive the recurrence. Error there compounds
  *multiplicatively along the sequence* instead of staying local to a position.
  At 0.09 % of Linear weight they are the cheapest possible place to spend
  precision.

The **large** SSM projections (`in_proj_qkv`, `in_proj_z`) *are* quantized. The
sibling recipe `qwen36-moe-35b-nvfp4` excludes everything matching
`linear_attn.*in_proj`, which here would park 4.03 B params (15.4 %) in bf16.
Those two are ordinary dense GEMMs whose outputs are consumed elementwise per
position; only `a` and `b` enter the recurrence itself. Splitting them raised
coverage from 79 % to 93 %.

Norms (`Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`) and the depthwise `conv1d` are
not `nn.Linear`, so `targets="Linear"` already excludes them.

The ignore list is derived from `model.named_modules()` at runtime, so it follows
the architecture rather than assuming layer indices.

### Validated against the official config

`reference/qwen35_official_fp8_config.json` holds the `modules_to_not_convert`
list from the official Qwen3.5-family FP8 config.

```bash
python verify_ignore.py \
    --model_path "$MODEL_PATH" \
    --reference  reference/qwen35_official_fp8_config.json
```

It walks the real module tree and classifies every reference entry:

```
reference entries          : 40
our ignore list            : 207 nn.Linear modules
[OK]  already covered by us: 9
[OK]  not an nn.Linear     : 10   (RMSNorm x4, LayerNorm, Conv3d, Conv1d,
                                   Embedding, RMSNormGated, the visual container)
[OK]  absent from this ckpt: 21   (mtp.*, mlp.gate/shared_expert_gate,
                                   deepstack_merger_list.*, in_proj_ba, A_log, dt_bias)
[GAP] Linear, unprotected  : 0
```

A raw set-diff between the two lists is misleading, because most reference
entries name modules that `targets="Linear"` never reaches. What matters is the
last row: **zero** reference exclusions are an `nn.Linear` present in this
checkpoint that we quantize. The script exits non-zero if that count is not zero.

The important agreement is on the judgment call above — the official config
protects `in_proj_a`, `in_proj_b` and the fused `in_proj_ba` while quantizing
`in_proj_qkv`, `in_proj_z` and `out_proj`. That is exactly the split derived
here, and the opposite of the blanket `linear_attn.*in_proj` heuristic.

`in_proj_ba`, `deepstack_merger_list`, the MoE gates and the MTP head are all
covered by the policy regexes even though this dense checkpoint does not
instantiate them, so the same table transfers to the MoE sibling.

## Strategies

| `--strategy` | llm-compressor scheme | weights | activations | size | hardware |
|--------------|-----------------------|---------|-------------|-----:|----------|
| `fp8` | `FP8` | 8-bit per-tensor | 8-bit per-tensor, static | 29 G | SM 89+ |
| `fp8-dynamic` | `FP8_DYNAMIC` | 8-bit **per-channel** | 8-bit **per-token**, dynamic | 29 G | SM 89+ |
| `fp8-block` | `FP8_BLOCK` | 8-bit, 128×128 blocks | 8-bit, group-128 dynamic | 29 G | SM 89+ |
| `nvfp4` | `NVFP4` | 4-bit, group-16 | 4-bit, group-16 | 19 G | SM 100+ |
| `nvfp4a16` | `NVFP4A16` | 4-bit, group-16 | bf16 | 19 G | SM 80+ |

(bf16 base: 52 G.) `fp8`, `fp8-dynamic` and `nvfp4` are the three published
builds; `fp8-block` and `nvfp4a16` are available but not part of the standard
build. `fp8-block` matches the official Qwen3.5-family FP8 config
(`weight_block_size [128,128]`, `activation_scheme dynamic`).

**On `fp8` vs `fp8-dynamic`:** `FP8` collapses both weights and activations to
one per-tensor scalar, fitted on the calibration set. `FP8_DYNAMIC` keeps
per-channel weight scales and computes activation scales per token at runtime —
generally the more accurate of the two, and what production FP8 checkpoints ship.
Both are built so the comparison is measurable rather than assumed.

**NVFP4 below SM 100:** NVFP4 kernels need Blackwell. The checkpoint produced on
an H100 is numerically valid and runs correctly there — it just does not get the
Blackwell speedup. `--allow_unsupported_gpu` acknowledges that; without it the
run stops with exit code 2.

## Pipeline

Every strategy runs `SmoothQuantModifier(smoothing_strength=0.8)` before the
quantizer, matching the `cosmos-reason2` recipe in this repository. That recipe
structures the work as two passes — pass 1 SmoothQuant + quantize the LLM, pass 2
quantize the ViT. Here the entire vision tower stays bf16, so pass 2 is a no-op
and only pass 1 runs.

SmoothQuant fits its migration scales from activations, so **every** strategy
calibrates on CNN/DailyMail, including the schemes whose quantization is
otherwise data-free.

### SmoothQuant mappings are built per layer, not as global regexes

The `cosmos-reason2` recipe expresses its mappings as two global regexes. That
does not survive here, for two reasons:

1. **Hybrid stack.** `input_layernorm` feeds `self_attn.{q,k,v}_proj` in the 16
   full-attention layers but `linear_attn.in_proj_{qkv,z,a,b}` in the other 48.
   One regex covering both makes llm-compressor 0.13's `match_modules_set`
   accumulate unmatched norms until it fails with *"SmoothQuant must match a
   single smooth layer for each mapping"*.
2. **Correctness.** SmoothQuant divides the norm's output scale and multiplies it
   back into the listed consumers. A consumer left off the list silently receives
   mis-scaled activations — so `in_proj_a` and `in_proj_b` must be listed even
   though they stay bf16.

`build_smoothquant_mappings()` therefore builds one mapping per (layer, norm)
pair from the live module tree using exact module names: **128 mappings = 16
full-attention + 48 linear-attention + 64 MLP**, each holding exactly one norm
and precisely the Linears that consume it.

### NVFP4 global scales are shared across vLLM's fused packs

NVFP4 carries one FP32 *global* scale per tensor, and every module vLLM packs
into a single kernel has to share it. vLLM's `packed_modules_mapping` for
`qwen3_5` fuses `in_proj_qkv` + `in_proj_z` into `in_proj_qkvz`, which
llm-compressor's `FUSED_LAYER_NAMES` does not know about. Left alone the two get
independent global scales (measured: 328.0 vs 472.0 on layer 0) and vLLM warns at
load that the weight global scale differs for parallel layers — 96 modules across
the 48 linear-attention layers, 15.4 % of all Linear weight, running through a
mis-scaled fused GEMM. `patch_fused_layer_names()` registers the pair before the
run. It is applied only for the FP4 schemes, which are the only ones with a
global scale.

## Calibration

CNN/DailyMail (`abisee/cnn_dailymail`, config `3.0.0`, split `train`, field
`article`), streamed and shuffled with `buffer_size = 3 × num_samples`, 512
samples at 2048 tokens, seed 42.

Text-only, deliberately: the vision tower is not being quantized, so text inputs
are the right calibration signal and the VLM processor is skipped entirely.

## Checkpoint layout

`save_pretrained` writes the quantized safetensors plus `config.json`, then every
other base-model file is copied verbatim:

```
chat_template.jinja  generation_config.json  merges.txt  preprocessor_config.json
tokenizer.json  tokenizer_config.json  video_preprocessor_config.json  vocab.json
LICENSE  .gitattributes
```

`processor.save_pretrained()` is deliberately **not** called: re-serializing the
processor rewrites `preprocessor_config.json` with whatever calibration left on
it, which breaks inference. `config.json` is post-processed to drop `zp_dtype`
and `scale_dtype`, which some runtimes reject. A `quantization_manifest.json`
records the strategy, scheme, SmoothQuant settings, calibration settings and the
resolved ignore patterns.

## Validation

Minimum checks after a run:

1. **The census matches the policy** — `inspect_model.py` reports 93.28 % of
   Linear weight covered and 207 modules held at bf16.
2. **No gaps against the official config** — `verify_ignore.py` exits 0 with
   `[GAP] Linear, unprotected : 0`.
3. **The manifest is written** — `quantization_manifest.json` exists in the
   output directory and records 128 SmoothQuant mappings.
4. **The checkpoint generates coherent text** — a broken quantization shows up as
   repetition or token soup long before any benchmark score moves:

   ```bash
   python sanity_gen.py --ckpt outputs/qwen38-27b-fp8
   ```

   Run this from the vLLM environment, not the quantization one.

## Measured accuracy

Measured with a separate harness — not part of this recipe — on the vLLM backend,
greedy decoding, thinking disabled, ERQA n=400 and RealWorldQA n=765.

| checkpoint | size | ERQA | RealWorldQA |
|------------|-----:|-----:|------------:|
| bf16 (base) | 52 G | 0.5100 | 0.7765 |
| `fp8` | 29 G | 0.4725 | 0.7778 |
| `fp8-dynamic` | 29 G | 0.4825 | 0.7830 |
| `nvfp4` | 19 G | 0.4325 | 0.7229 |

Δ against bf16, with a two-proportion z-test:

| checkpoint | ERQA | RealWorldQA |
|------------|------|-------------|
| `fp8` | −3.75 pt (p=0.288, ns) | +0.13 pt (p=0.951, ns) |
| `fp8-dynamic` | −2.75 pt (p=0.436, ns) | +0.65 pt (p=0.758, ns) |
| `nvfp4` | −7.75 pt (p=0.028) **significant** | −5.36 pt (p=0.015) **significant** |

**Both FP8 variants hold the baseline** — neither drop is distinguishable from
noise at this sample size, on either task. `fp8-dynamic` is nominally ahead of
`fp8` on both, consistent with per-channel plus per-token scales beating
per-tensor, but that gap is itself within noise. Either is a safe 52 G → 29 G
swap.

**NVFP4 costs real accuracy** — the only checkpoint whose drop is significant,
and it is significant on both tasks independently. 19 G is a 2.7× reduction, so
it remains the right pick when memory is the binding constraint, but it is not a
free swap the way FP8 is.

Read `ns` as "no measurable drop at n=400/765", not "identical". Separating a
one-point difference would need roughly 10× the documents. Absolute scores sit
below Qwen's published numbers because thinking is disabled here; that does not
affect the comparison, since all four rows use the identical protocol and the
bf16 row is the reference every delta is measured against.

## Gotchas this model hits

Recorded because each one cost a run. All are already handled in the scripts.

1. **`save_original_format=False` on save.** llm-compressor's sequential pipeline
   leaves modules CPU-offloaded, and transformers ≥ 5 otherwise tries to revert
   its weight conversions at save time, failing with *"could not revert some
   weight conversions because of offloading"*. `quantize.py` probes the
   unwrapped `transformers.PreTrainedModel.save_pretrained` signature before
   passing the flag, because `model.save_pretrained` is llm-compressor's wrapper
   and only exposes `**kwargs`.
2. **The model class must come from `config.architectures`.** Going through
   `AutoModelForCausalLM` silently yields `Qwen3_5ForCausalLM` — the text stack
   only — which hides the vision tower from both the census and the ignore list.
3. **Attention metadata lives under `text_config`.** llm-compressor reads
   `num_attention_heads` and friends off the top-level config; `quantize.py`
   mirrors them up so the KV bookkeeping resolves.
4. **`max_num_seqs` when serving.** Every decode sequence in a Gated DeltaNet
   layer holds one Mamba-style state block, so concurrency is bounded by state
   memory rather than by the KV cache. vLLM's default of 1024 aborts startup with
   *"max_num_seqs (1024) exceeds available Mamba cache blocks"*. `sanity_gen.py`
   defaults to 64.
5. **The CUDA toolkit must match the torch build.** vLLM and flashinfer
   JIT-compile against `$CUDA_HOME` at *runtime*, so a mismatch does not fail at
   import — it fails much later as `undefined symbol: ..., version
   libcudart.so.12`.

## Published checkpoints

| Checkpoint | Format | Size |
|------------|--------|-----:|
| [`vrfai/Qwen3.8-27B-FP8`](https://huggingface.co/vrfai/Qwen3.8-27B-FP8) | FP8 W8A8, static per-tensor | 29 G |
| [`vrfai/Qwen3.8-27B-FP8-dynamic`](https://huggingface.co/vrfai/Qwen3.8-27B-FP8-dynamic) | FP8 W8A8, per-channel + per-token dynamic | 29 G |
| [`vrfai/Qwen3.8-27B-NVFP4`](https://huggingface.co/vrfai/Qwen3.8-27B-NVFP4) | NVFP4 W4A4, group-16 | 19 G |

## Authors

- VinRobotics AI Team
- Hugging Face: [vrfai](https://huggingface.co/vrfai)

## Credits

- Original model: [Qwen Team](https://huggingface.co/Qwen) (Alibaba Group)
- Quantization framework: [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
- Calibration data: [`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail)
