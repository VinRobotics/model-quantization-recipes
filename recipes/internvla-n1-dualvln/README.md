# internvla-n1-dualvln

FP8 quantization and TensorRT-Edge-LLM deployment for InternVLA-N1-DualVLN, a dual-system
vision-language navigation model, targeting NVIDIA Jetson Thor.

## Status

- Owner: unassigned
- Model family: `InternVLA-N1-DualVLN` (System 2 = Qwen2.5-VL-7B, System 1 = NextDiT trajectory head)
- Quantization presets: `fp8_default`, `fp8_per_channel` (validated) · `nvfp4_*` (experimental, see Notes)
- Runtime target: TensorRT-Edge-LLM engines on Jetson Thor (sm_110)

## What this model is, and what the metric has to be

InternVLA-N1 is a **dual-system** navigation policy:

    camera image ─► ViT (BF16/FP8 TRT) ─► LLM (FP16/FP8 TRT) ─► branch
                                                        ├─ discrete action (↑ ← → STOP)
                                                        └─ pixel goal (coordinate)
       System 2 (planner, ~2 Hz)                              ▼
       ────────────────────────────  latent_queries ─► LLM hidden ─► norm ─► cond_projector
                                                                                  │ z_latents
       System 1 (controller, ~15 Hz)                                             ▼
       ────────────────────────────  RGB ─► memory block (BF16 TRT) ─┐
                                                                     ├─► traj_dit (BF16 TRT, x10)
                                                        z_latents ───┘

Only **System 2** is quantized. System 1 stays BF16.

The two are joined by `z_latents`: the last-layer hidden states of 4 trajectory tokens, run
through a host-side `final_norm` and `cond_projector`. **That bridge is the acceptance
metric, not text quality.** A quantized checkpoint can emit perfectly fluent captions and
still be useless for navigation — NVFP4 does exactly that here (see Notes). Every claim in
this recipe is gated on `z_latents` cosine against the FP32 reference, not on generated text.

## Results

### Task accuracy — does the quantized model still pick the same waypoint?

Measured with `quantize/benchmark_accuracy.py` on 42 held-out samples from two scenes,
identical samples and greedy decoding, PyTorch on Jetson Thor:

| Checkpoint | pixel_goal_l2 mean | median | parse rate |
|---|---|---|---|
| System 2, BF16 (unquantized) | 47.24 px | **27.05 px** | 100 % |
| System 2, FP8 (s1) | 50.12 px | **26.97 px** | 100 % |

31 of 42 replies are byte-identical and the **median deviation between the two is 0.00 px**.
The 2.88 px gap in the mean is carried by 8 samples, worst case 89 px — it is a small number
of disagreements, not a systematic shift, which is why both statistics are reported. For
reference, the source project's self-validation gate for this metric is a median under 60 px.

Caveat worth keeping in view: this is a **weight-quantization** measurement. FP8 W8A8 also
quantizes activations, which the TensorRT engine does and the PyTorch path here does not, so
treat it as a lower bound on the engine's deviation rather than a prediction of it.

The official InternVLA-N1 metrics (SR, SPL, NE, OS, nDTW) are all closed-loop and need
Habitat or InternUtopia plus MP3D scenes. Neither is installed here and neither is practical
on a Jetson, so the published table (DualVLN: NE 4.05 / SR 64.3 / SPL 58.5 on VLN-CE R2R) is
a literature reference point, not something this recipe reproduces.

### Engine size and latency

Both engines built from the same repackaged System 2 and measured with `llm_bench` on
Jetson Thor, batch 1:

| | LLM engine | visual engine | prefill (1024 tok) | decode (pastKV 1024) |
|---|---|---|---|---|
| base FP16 (unquantized) | 15.0 GB | 1.36 GB | 196.22 ms | 86.16 ms |
| FP8 (s1) | **7.62 GB** | 1.36 GB | **95.80 ms** | **37.35 ms** |
| FP8 gain | 1.97x smaller | — | **2.05x faster** | **2.31x faster** |

Both produce the same text on the same prompt, so this is a straight win: FP8 halves the
engine and roughly doubles throughput while leaving the median waypoint error unchanged.

The FP16 engine is only correct because the build applies
`__LUNOWUD=-peep:fc_h_fusion=off`. The build log confirms it
(`Using __LUNOWUD=-peep:fc_h_fusion=off -peep:match_dual_gemm=off`). Without it, TensorRT
10.13 miscompiles Myelin's horizontal gate/up fusion on sm_110 and the engine emits fluent
gibberish — see the note further down.

### Earlier full-pipeline figures

Measured on Jetson Thor, 12 held-out multi-image VLN steps:

| LLM variant | z_latents vs FP32 | agrees w/ PyTorch | System 2 latency | LLM engine |
|---|---|---|---|---|
| PyTorch BF16 (baseline) | 0.99974 | — | 1631 ms · 1.00x | ~14 GB weights |
| base FP16 TensorRT (no quant) | **0.99985** | 12/12 | 770 ms · 2.12x | 14.2 GB |
| FP8 TensorRT | 0.99559 | 11/12 | **646 ms · 2.53x** | **7.6 GB** |
| NVFP4 TensorRT | **0.647** ✗ | tokens fine, bridge broken | — | 4.5 GB |

Other engines: ViT 1.3 GB BF16 → 0.68 GB FP8; traj_dit 0.07 GB; memory block 0.11 GB.

**Calibration data made no measurable difference.** Held-out z_latents came out at 0.99143
with generic `cnn_dailymail` text versus 0.99146 with a domain-specific VLN set — equal
within noise. An earlier apparent gain turned out to be overlap between the calibration and
probe sets. The VLN calibration loader ships anyway (it is the honest default to offer for a
navigation model), but do not expect it to buy accuracy.

## Files

    .
    ├── README.md
    ├── Makefile                          # command panel for both paths
    ├── requirements.txt                  # PyPI dependencies (excluding PyTorch)
    ├── requirements-torch.txt            # PyTorch installation guide
    ├── configs/
    │   └── schemes.yaml                  # scheme x strategy validity matrix
    ├── quantize/                         # HF checkpoint -> quantized HF checkpoint
    │   ├── README.md
    │   ├── repackage_system2.py          # strip System 1 -> stock Qwen2.5-VL checkpoint
    │   ├── quantize.py                   # ModelOpt driver
    │   ├── configs.py                    # presets, strategies, validity gate
    │   ├── calibration.py                # text / multimodal / VLN calibration loaders
    │   ├── model.py                      # load, calibrate, export
    │   ├── prompt_builder.py             # VLN prompt — single source of truth
    │   └── scripts/{00_fetch_calib_scenes,01_repackage,02_quantize}.sh
    └── trt-edgellm/                      # quantized checkpoint -> engines -> verification
        ├── README.md
        ├── export_traj_dit.py            # System 1 diffusion head -> ONNX -> BF16 engine
        ├── export_memory_block.py        # System 1 memory block -> ONNX -> BF16 engine
        ├── engine_runner.py              # direct-TensorRT LLM harness (hand-built 3D mRoPE)
        ├── internvla_compat.py           # patches needed to load System 1
        ├── investigate_nvfp4.py          # why z_latents collapse under NVFP4
        ├── verify/                       # 7 fidelity checks
        ├── benchmark/                    # 3 latency/memory benchmarks
        ├── deploy/run_eval_engine.py
        └── scripts/{03_export_build_system2,04_export_system1,05_verify,06_benchmark}.sh

## The repackage step, and why it matters

`InternVLA-N1-DualVLN` declares `model_type: internvla_n1`, ships **no** modeling code in the
checkpoint, and therefore cannot be loaded with `trust_remote_code` — the class has to come
from the InternNav repository.

`quantize/repackage_system2.py` sidesteps that for the entire quantization flow. It is pure
safetensors surgery: it streams the checkpoint, drops the eight System-1 tensor prefixes,
and rewrites `config.json` to `model_type: qwen2_5_vl` /
`architectures: ["Qwen2_5_VLForConditionalGeneration"]`. **It imports nothing from
InternNav.**

After that step everything downstream is a stock Qwen2.5-VL flow:

| Step | Needs `INTERNNAV_PATH`? |
|---|---|
| `00_fetch_calib_scenes.sh` | no |
| `01_repackage.sh` | **no** — pure file surgery |
| `02_quantize.sh` | no — operates on stock Qwen2.5-VL |
| `03_export_build_system2.sh` | no |
| `04_export_system1.sh` | **yes** |
| `05_verify.sh` (latents) | no |
| `05_verify.sh` (agent-level) | **yes** |

Cost of this approach is one ~15 GB intermediate checkpoint. Pass `--free_source` to delete
each input shard as its converted copy is written if disk is tight.

## Setup

**Step 1 — PyTorch.** On Jetson, use the JetPack wheel; do not install from PyPI.

| Platform | Command |
|---|---|
| Jetson Thor (JetPack 7.1, CUDA 13.0) | use the JetPack-provided `torch==2.10.0`; see requirements-torch.txt |
| x86 CUDA 12.8 | `pip install torch==2.10.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128` |

**Step 2 — remaining dependencies:**

    pip install -r requirements.txt

**Step 2b — pin numpy afterwards.** This recipe needs numpy 1.x (OpenCV and diffusers break
under numpy 2 on Jetson), while the other recipes in this repository pin `numpy==2.2.6`.
Since `uv lock` resolves every extra into one universal lock, declaring both would make the
lock unsatisfiable, so numpy is deliberately absent from this recipe's extra. Run once after
`uv sync`:

    pip install "numpy==1.26.4" "scipy==1.13.1"

**Step 3 — dependencies that are not on PyPI.** These must be present before running anything:

| Dependency | How |
|---|---|
| TensorRT 10.13 | ships with JetPack at `/usr/lib/python3.12/dist-packages` |
| `tensorrt-edgellm` 0.8.0 | build from source, then `pip install --no-deps -e $TRT_EDGE_LLM` |
| InternNav | `git clone` it; export `INTERNNAV_PATH`. Only needed for System 1 and agent-level checks |
| OpenCV | system package |

## Environment

    export INTERNVLA_CKPT=/path/to/InternVLA-N1-DualVLN   # source checkpoint
    export INTERNNAV_PATH=/path/to/InternNav              # System 1 only
    export TRT_EDGE_LLM=/path/to/TensorRT-Edge-LLM        # build root
    export VLN_OPT_WORK=$HOME/vln-opt-work                # intermediate artifacts
    export VLN_OPT_ENGINES=$VLN_OPT_WORK/engines          # engine output

Two flags are mandatory and set by the scripts themselves — listed here so their absence is
diagnosable, not so you set them by hand:

- `TRITON_BACKENDS_IN_TREE=1` for every quantization run.
- `__LUNOWUD="-peep:fc_h_fusion=off"` for every engine build on TensorRT 10.13/10.14. Without
  it the FP16 engine emits gibberish — a Myelin miscompile on sm_110, not a precision problem.

## Quick Start

    make repackage          # InternVLA checkpoint -> stock Qwen2.5-VL System 2
    make quantize-fp8       # s1 FP8, text calibration
    make export-build       # ONNX export + FP8 LLM engine + visual engine
    make verify-latents     # the acceptance gate: z_latents cosine > 0.99

`make help` lists every target. Each script is also runnable directly; see the per-path
READMEs in `quantize/` and `trt-edgellm/`.

## CLI Reference

    quantize/quantize.py [options]

      --model_path PATH           Repackaged System 2 checkpoint (required)
      --output_path PATH          Destination for the quantized checkpoint (required)
      --strategy {s1,s2,s3,s4}    s1 LLM · s2 +KV cache · s3 +ViT · s4 +ViT+KV (default: s1)
      --scheme NAME               Preset from configs/schemes.yaml (default: fp8_default)
      --calib {auto,text,multimodal,vln}   Calibration source (default: auto)
      --calib_data PATH           Root for VLN calibration scenes
      --num_calib_samples N       Calibration samples (default: 512; image paths cap at 128)
      --max_seq_len N             Calibration truncation length (default: 512)
      --dtype {fp16,bf16}         Load dtype (default: bf16)
      --device DEVICE             Torch device (default: cuda)
      --resume DIR                Layerwise checkpoint dir, for AWQ/Hessian crash recovery
      --allow_experimental        Permit schemes marked experimental (NVFP4)
      --dry_run                   Load and validate, skip quantization

    quantize/repackage_system2.py [options]

      --model_path PATH           Source InternVLA-N1-DualVLN checkpoint (required)
      --output_path PATH          Destination stock Qwen2.5-VL checkpoint (required)
      --free_source               Delete each source shard once converted (destructive)

## Notes

### What is and is not quantized

Quantized: the System 2 LLM backbone (`model.layers.*`), and the vision tower under `s3`/`s4`.
Never quantized: System 1 (`traj_dit`, memory block), `lm_head`, and the host-side bridge ops
(`final_norm`, `cond_projector`).

### Scheme validity

|  | s1 (LLM) | s2 (+KV) | s3 (+ViT) | s4 (+ViT+KV) |
|---|---|---|---|---|
| `fp8_default`, `fp8_per_channel` | yes | yes | yes | yes |
| `nvfp4_*` | experimental | experimental | **blocked** | **blocked** |

- **NVFP4 with the vision tower is blocked, permanently.** The Qwen2.5-VL ViT MLP has
  `intermediate_size = 3420`, and 3420 / 16 = 213.75 — not divisible by the NVFP4 block size.
  The recipe rejects these combinations up front rather than letting them crash mid-run.
- **KV cache is always FP8**, even with NVFP4 weights. NVFP4 KV requires `sm100f` (datacenter
  Blackwell); Thor is sm110. This is a hardware limit, not a configuration mistake — do not
  "fix" it by selecting an NVFP4 KV preset.
- **NVFP4 weights are experimental and currently not usable for navigation.** They quantize,
  export and generate fluent text, but `z_latents` cosine falls to 0.647, which breaks the
  System 2 → System 1 bridge. Requires `--allow_experimental`. See
  `trt-edgellm/investigate_nvfp4.py`.

### FP16 on Thor is fine — an earlier claim to the contrary was wrong

Earlier notes in the source project stated that a plain FP16 LLM engine "produces garbage on
Thor, so FP8 is mandatory". **That is incorrect and has been retracted.** The garbage came
from a Myelin `fc_h_fusion` miscompile on sm_110 at TensorRT 10.13: the horizontal fusion of
the gate/up projections is wrong at batch 1. TensorRT-Edge-LLM already disables it, but only
for TensorRT >= 10.15, so 10.13 slips through the gap. FP8 quietly dodged the same bug
because its Q/DQ nodes break the fusion pattern — which is why FP8 *looked* mandatory.

Exporting `__LUNOWUD="-peep:fc_h_fusion=off"` fixes FP16 completely; the base FP16 engine is
the highest-fidelity variant in the results table. FP8 remains the recommended deployment
choice on size and latency, not on correctness.

### One visual engine, sized for multi-image prompts

The visual encoder is built once at `--minImageTokens 4 --maxImageTokens 4096
--maxImageTokensPerImage 1024`, and every consumer reads that one engine.

This is worth stating because the source project got it wrong in a way that is easy to
repeat: its default build used `128 / 512 / 512`, taken from a single-image demo. A VLN
prompt carries 9–10 images and roughly 1,764 image tokens, so it does not fit in 512, and the
multi-image verification scripts were quietly pointed at a hand-built engine that no script
in the repository produced. If a verification here reports a shape or capacity error, check
the visual engine's sizing before suspecting the weights.

### Scope

These are **conversion-fidelity** numbers plus offline planner metrics. They are not a
closed-loop navigation success rate — that needs the Habitat/InternUtopia simulator, which is
not part of this recipe. `trt-edgellm/verify/verify_engine_policy.py` checks that the
simulator adapter is wired correctly, but cannot exercise it.

## Tested Environments

- **OS:** Ubuntu 24.04 (JetPack 7.1)
- **Hardware:** NVIDIA Jetson Thor (Blackwell, sm_110, 128 GB unified memory)
- **Python:** 3.12.3
- **PyTorch:** 2.10.0
- **CUDA:** 13.0
- **TensorRT:** 10.13.3.9
- **tensorrt-edgellm:** 0.8.0
- **nvidia-modelopt:** 0.44.0
- **transformers:** 4.51.3 · **diffusers:** 0.33.1 · **onnx:** 1.22.0 · **numpy:** 1.26.4
