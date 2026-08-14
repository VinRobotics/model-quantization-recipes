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

### Benchmark matrix

All three variants built from the same repackaged System 2 and measured on Jetson Thor,
batch 1, on an idle GPU.

| Variant | checkpoint | LLM engine | visual | prefill (1024) | decode (pastKV 1024) | z_latents (engine) | z_latents (weights only) | pixel L2 mean / median |
|---|---|---|---|---|---|---|---|---|
| BF16 (unquantized) | 16.6 GB | 14.15 GB | 1.36 GB | 135.8 ms | 56.4 ms | **0.999471** | — | 47.24 / 27.05 px |
| **FP8 s1** | 10.1 GB | **7.62 GB** | 1.36 GB | **82.1 ms** | **31.5 ms** | **0.991861** | 0.998020 | 46.26 / **22.51 px** |
| NVFP4 s1 (experimental) | 7.2 GB | 4.77 GB | 1.36 GB | 73.2 ms | 20.2 ms | 0.931005 ✗ | 0.987986 | 40.69 / 23.54 px |

Weight quantization error, mean relative over 21 projections in layers 0/13/27:
FP8 **2.67 %**, NVFP4 **9.45 %**.

**FP8 is the recommended scheme.** Against BF16 it is 1.86x smaller and 1.65x/1.79x faster,
holds the bridge at 0.9919, and the median waypoint error does not get worse — it improves
slightly (27.05 → 22.51 px), which is within the spread of a 42-sample set and should be read
as "unchanged", not as a gain from quantization.

**NVFP4 is faster and smaller still but fails the gate.** Its bridge sits at 0.931, below the
0.99 threshold, so it is not recommended for navigation despite the attractive size and
latency. See the NVFP4 section for where that number comes from.

### Where NVFP4's loss actually comes from

Three measurements per scheme, each isolating one layer. The middle one — PyTorch with live
quantizers, so weights *and* activations are simulated exactly as the engine does them — is
what makes this decomposable:

| | weights only | fake quant (W4A4) | engine |
|---|---|---|---|
| what is quantized | weights | weights + activations | weights + activations, real kernels |
| FP16 | — | — | 0.999471 |
| FP8 | 0.998020 | — | 0.991861 |
| **NVFP4** | **0.987986** | **0.978631** | **0.931005** |

For NVFP4 that splits the 0.057 total loss as:

- weight quantization: 0.012
- activation quantization: **0.009**
- **everything else, inside the engine: 0.048**

**This corrects an earlier claim in this file.** The gap was previously attributed to
activation quantization. It is not: activations account for 16 % of it, and 84 % appears
only once the model runs as a TensorRT engine. FP8 shows nothing comparable — its entire
PyTorch-to-engine gap is 0.006, while NVFP4 loses 0.048 there, nearly eight times more.

The FP16 engine at 0.999471 bounds TensorRT's generic cost at about 0.0005, so this is not
export or runtime overhead in general — it is specific to the NVFP4 path. That is the same
neighbourhood as the known CASK epilogue miscompile, which `-cask_fusion:max_num_epilogues=1`
already improves from 0.647 to 0.931 but evidently does not fully resolve.

#### Where the engine loses it — measured, not inferred

`trt-edgellm/diagnose_engine_gap.py` compares the engine against the **fake-quant** model
rather than against the unquantized one, in a single process on the same inputs, so the
difference is the engine alone. On a text prompt, hidden states before the final norm:

| engine | vs its own fake quant |
|---|---|
| FP8 | **0.998256** |
| NVFP4, `maxBatchSize 1` + CASK cap | **0.986790** |
| NVFP4, `maxBatchSize 2`, no CASK flag | **0.986790** |

Two things follow.

**The NVFP4 engine carries about eight times FP8's engine-side error** — 0.013 against
0.002 — and the bridge amplifies it: a 0.013 hidden-state deviation becomes the 0.048 seen
in z_latents once it passes through the final norm, GELU and `cond_projector`.

**It is not the batch-1 miscompile.** The two NVFP4 engines are genuinely different builds
(different checksums, `maxBatchSize` 1 and 2, and the fork correctly withholds
`-cask_fusion:max_num_epilogues=1` from the batch-2 build) and they measure identically to
six decimal places. So `max_num_epilogues=1` fully recovers whatever the batch-1 path loses,
and the residual deficit is inherent to the NVFP4 kernels, independent of batch size. The
earlier guess that a second batch-1 miscompile was hiding here is wrong.

A harness bug found this section's control worth having: comparing against
`output_hidden_states[-1]` reads 0.49 for a *known-good* FP8 engine, because the engine
emits hidden states before the final norm while that tensor is after it. The FP8 control
caught it; without one, 0.48 for NVFP4 would have looked like a broken kernel.

**Practical consequence:** NVFP4 on this model is better than its engine number suggests.
At 0.9786 the quantization itself is close to the 0.99 gate. Anyone wanting to make NVFP4
viable here should look at the TensorRT NVFP4 kernel path, not at better quantization
algorithms — which is also why AWQ, local-Hessian and QAT all failed to move it.

### Two things to know before reading these numbers

**Run everything on an idle GPU.** Latency measured while another job shared the device came
out 40-60 % higher (FP8 prefill 117 ms against 82 ms). The measurement script does not
enforce this.

**The "identical replies" count is a weak indicator.** It moved from 31/42 to 11/42 for FP8
across code revisions of the checkpoint loader, while `pixel_goal_l2` barely changed. Greedy
decoding over a 152k vocabulary flips on tiny logit differences, so treat the L2 median as the
number that matters and the identity count as colour.

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

### Bridge fidelity — z_latents

`trt-edgellm/verify/verify_latents.py` reconstructs the System 2 -> System 1 bridge against
a PyTorch BF16 reference: embed, scatter image embeddings, append the 4 trajectory tokens,
run the engine with hand-built 3D mRoPE, take the last-layer hidden states, then apply the
host-side norm and `cond_projector`.

| Engine | hidden pre-norm | hidden post-norm | **z_latents** | rel-L2 |
|---|---|---|---|---|
| base FP16 | 0.999843 | 0.999123 | **0.999471** | 0.0293 |
| FP8 (s1) | 0.997793 | 0.987343 | **0.991861** | 0.1023 |

Both pass the > 0.99 gate. This is the check that NVFP4 fails (0.647), and it is the reason
it ships as experimental: text stays fluent while the waypoint bridge collapses.

### System 1 engines (BF16, not quantized)

| Engine | ONNX | engine | I/O |
|---|---|---|---|
| traj_dit | 134 MB | **72 MB** | `x`, `timestep`, `z_latents` -> `output` |
| memory block | 200 MB | **104 MB** | `images` -> `memory_tokens` |

Both are BF16 via `trtexec` and deliberately stay unquantized: they are small enough that
quantizing them buys nothing, and the diffusion head is the part least tolerant of it.

Note the environment split. Exporting System 1 needs transformers 4.51.3 (Python 3.10 here),
while the TensorRT Python bindings ship for Python 3.12. The exporters therefore build the
engine and skip their in-script parity check with a message rather than failing; run
`verify/verify_system1.py` under the 3.12 environment to check parity.

### Verification inventory

| Check | Needs | Status here |
|---|---|---|
| `verify_latents.py` | engine + bridge tensors | **run** — FP16 0.9995, FP8 0.9919 |
| `verify_latents_vln.py` | engine + held-out VLN episodes | ready |
| `verify_engine_policy.py` | engine + `INTERNNAV_PATH` | **run** — PASS 2/2 |
| `verify_system1.py` | System-1 engines + `INTERNNAV_PATH` | ready |
| `verify_accuracy.py` | engine + `INTERNNAV_PATH` + agent assets | ready |
| `verify_pixelgoal_gt.py` | engine + held-out parquet ground truth | ready |
| `verify_e2e_agent.py` | all engines + `INTERNNAV_PATH` | ready |
| `benchmark/benchmark_system2.py` | a **user-supplied** golden manifest | needs input |
| `benchmark/bench_system1.py`, `bench_memory.py` | `INTERNNAV_PATH` | ready |

`benchmark_system2.py` needs a golden manifest that nothing in this recipe (or the source
project) generates; it now fails with an explanation of the expected file shape rather than
a bare `FileNotFoundError` at the first read.

### NVFP4 — the collapse had two causes, and one of them was the compiler

`trt-edgellm/investigate_nvfp4.py` was written to explain why NVFP4 keeps text fluent while
the bridge collapses. Three measurements, each isolating a different layer:

| | weight rel-err | z_latents |
|---|---|---|
| FP8, weights only (PyTorch) | 2.67 % | 0.998020 |
| **NVFP4, weights only (PyTorch)** | 9.45 % | **0.987986** |
| **NVFP4 engine, with the CASK workaround** | — | **0.931005** |
| NVFP4 engine, no workaround (source project's figure) | — | 0.647 |

Reading them together:

* **Weight quantization is not the problem.** NVFP4 weights cost 0.988 — error 3.5x FP8's,
  with the bridge degrading roughly in proportion. Nothing anomalous.
* **Most of the old 0.647 was a compiler artifact.** Rebuilding the engine with
  `-cask_fusion:max_num_epilogues=1` (which the fork applies automatically, gated to NVFP4
  graphs at batch 1) moves it to 0.931. The build log confirms all three flags fired:
  `-peep:fc_h_fusion=off -peep:match_dual_gemm=off -cask_fusion:max_num_epilogues=1`.
* **A real gap remains.** 0.988 weights-only versus 0.931 through the engine is the part
  this platform's PyTorch path cannot model: NVFP4 is W4A4, and 4-bit *activations* through
  a 3584-wide hidden state in blocks of 16 are the remaining suspect.

The 0.647 figure is quoted from the source project and was not reproduced here; what is
measured here is that the same checkpoint reaches 0.931 once the workaround is applied.

Channel analysis rules out the obvious remedy for the weight-side loss: at the final layer
the top 128 channels by magnitude carry only 29.5 % of the squared error, and masking them
*lowers* cosine rather than restoring it. The error is spread, not concentrated in outliers,
so AWQ scaling or a targeted exclusion has nothing to grip.

**No post-training method closes the weight-side gap.** All three NVFP4 presets land in the
same place, measured end to end against the unquantized reference:

| Preset | z_latents (weights) | note |
|---|---|---|
| `nvfp4_default` | 0.987986 | baseline |
| `nvfp4_awq_full` | 0.986293 | equal within noise |
| `nvfp4_local_hessian` | 0.987986 | **byte-identical to default** |

`nvfp4_local_hessian` is a no-op on this model. Its preset genuinely differs
(`algorithm={'method': 'local_hessian', 'fp8_scale_sweep': True}` against `'max'`) and the
run exits 0, but the exported weights match `nvfp4_default` bit for bit — 0 of 6,422,528
bytes differ, scales included. Two quantization runs with different algorithms cannot
produce identical output unless the algorithm did not run. It remains selectable, so a user
would reasonably believe they had tried it.

AWQ does run — its weights genuinely differ — but does not help, which is what the channel
analysis predicted: the error is spread rather than carried by outliers, so per-channel
rescaling has nothing to grip.

That leaves quantization-aware training as the only remaining lever, since the gap between
0.988 (weights) and 0.931 (engine) is 4-bit activations and no post-training method reaches
those. See `quantize/qat.py`.

### QAT was tried and made it worse — but the run did not converge

One exploratory run: 64 samples, 16 optimizer steps, lr 1e-5, the last 4 decoder layers
trainable (932 M parameters), the rest frozen.

| | z_latents (engine) | pixel L2 mean / median |
|---|---|---|
| NVFP4 PTQ | **0.931005** | 40.69 / 23.54 px |
| NVFP4 + QAT | **0.891583** | 41.86 / 23.16 px |

Worse on the bridge, unchanged on the task within noise. That is consistent with the
training loss, which *rose* from 0.78 to 1.13 across the 16 steps — the run pushed the
weights in the wrong direction rather than converging.

**Read this as a failed training run, not as evidence that QAT cannot work here.** The loss
never fell, so the experiment never reached the question it was meant to answer. What would
change next: a much smaller learning rate (1e-6 or below — QAT adapts to quantization noise,
it does not relearn the task, and 1e-5 over 932 M parameters is too large a step), several
hundred steps rather than sixteen, and a warmup instead of a flat schedule.

Two practical notes for anyone repeating this:

Full fine-tuning does not fit. At 8.29 B parameters, weights plus gradients plus AdamW
moments come to ~100 GB before any activations, against a 122 GB pool shared with the host —
the first attempt was killed by the OOM killer. `--train_last_n_layers` (default 4) is what
makes it fit, and it also targets the right place: the bridge reads the last layer's hidden
states.

Freezing plus gradient checkpointing needs both fixes at once. The activations entering the
first trainable layer carry no `grad_fn`, and reentrant checkpointing then discards the
graph — `loss.backward()` fails with "element 0 of tensors does not require grad". Setting
`use_reentrant=False` **and** calling `enable_input_require_grads()` is required; either
alone still fails.

Finally, note that `z_latents` against the unquantized reference is the wrong metric for
QAT and is only reported here through the engine. PTQ tries to approximate the original
model, so similarity to it is meaningful. QAT deliberately moves the weights away from the
original to compensate for quantization noise, so a successful QAT run can *lower* that
similarity while improving real behaviour. Judge QAT on the engine and on task accuracy.

**Verdict: NVFP4 stays experimental.** 0.931 is below the 0.99 gate, so it is not
recommended for navigation — but it is far from the broken 0.647 it appeared to be, and the
remaining gap now has a named suspect rather than a mystery.

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
    │   ├── repackage_system2.py          # strip System 1 -> stock Qwen2.5-VL checkpoint
    │   ├── quantize.py                   # ModelOpt driver
    │   ├── quant_schemes.py              # scheme registry + validity gate
    │   ├── calibration.py                # text / multimodal / VLN calibration loaders
    │   ├── model_loader.py               # load, calibrate, export
    │   ├── load_quantized.py             # read a ModelOpt checkpoint back WITH its scales
    │   ├── benchmark_accuracy.py         # pixel-goal L2 on held-out VLN episodes
    │   ├── prompt_builder.py             # VLN prompt — single source of truth
    │   └── scripts/                      # 00_fetch_calib_scenes, 01_repackage, 02_quantize
    └── trt-edgellm/                      # quantized checkpoint -> engines -> verification
        ├── engine_runner.py              # direct-TensorRT LLM harness (hand-built 3D mRoPE)
        ├── export_traj_dit.py            # System 1 diffusion head -> ONNX -> BF16 engine
        ├── export_memory_block.py        # System 1 memory block -> ONNX -> BF16 engine
        ├── internvla_compat.py           # the three patches needed to load System 1
        ├── traj_dit_loader.py  memblock.py
        ├── trt_torch.py                  # NVIDIA Apache-2.0 — header kept, not restamped
        ├── engine_policy.py              # simulator adapter (untested: needs Habitat)
        ├── investigate_nvfp4.py          # why NVFP4 breaks the System 1 bridge
        ├── verify/                       # 7 fidelity checks
        ├── benchmark/                    # 3 latency and memory benchmarks
        ├── deploy/run_eval_engine.py
        └── scripts/03_export_build_system2.sh

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

**Step 2c — System 1 needs a different transformers.** The InternNav modeling code reads
`config.hidden_size` off the top-level config, which transformers 5.x no longer flattens, so
System-1 export fails there with `'InternVLAN1ModelConfig' object has no attribute
'hidden_size'`. Run the System-1 steps under **transformers 4.51.3** with `diffusers==0.33.1`
and `onnx==1.22.0` present. The System-2 path (repackage, quantize, export, engine build,
latent verification) is unaffected and runs on either.

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
