# internvla-n1-dualvln

FP8 quantization and TensorRT-Edge-LLM deployment for InternVLA-N1-DualVLN, a dual-system
vision-language navigation model, targeting NVIDIA Jetson Thor.

## Status

- Model family: `InternVLA-N1-DualVLN` (System 2 = Qwen2.5-VL-7B, System 1 = NextDiT trajectory head)
- Quantization presets: `fp8_default`, `fp8_per_channel` (validated) · `nvfp4_*` (experimental, see Notes)
- Runtime target: TensorRT-Edge-LLM engines on Jetson Thor (sm_110)
- End to end on TensorRT: System 2 FP8 **and** System 1 BF16, both verified against PyTorch
  (bridge 0.9919, System-1 trajectory 0.9997) — **2.55x** faster per planning step at
  9.16 GB of weights against 15.7 GB unquantized

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

### Benchmark matrix — both systems

Everything below was measured on one Jetson Thor, batch 1, on an idle GPU. **System 2 is the
only part that gets quantized**; System 1 stays BF16 by design, so its row is a conversion
result rather than a quantization one.

#### System 2 — Qwen2.5-VL-7B planner (quantized)

| Variant | weights / activations | KV cache | vision tower | checkpoint | LLM engine | visual engine | prefill (1024) | decode (pastKV 1024) | z_latents (engine) | pixel L2 mean / median |
|---|---|---|---|---|---|---|---|---|---|---|
| BF16 baseline | BF16 W16A16 -> FP16 engine | FP16 | BF16 | 16.6 GB | 14.15 GB | 1.36 GB | 135.8 ms | 56.4 ms | **0.999471** | 47.24 / 27.05 px |
| **FP8 s1** | **FP8 E4M3, W8A8, per-channel** | FP16 | BF16 | 10.1 GB | **7.62 GB** | 1.36 GB | **82.1 ms** | **31.5 ms** | **0.991861** | 46.26 / **22.51 px** |
| NVFP4 s1 (experimental) | NVFP4 E2M1, W4A4, block 16 w/ FP8 block scales | FP16 | BF16 | 7.2 GB | 4.77 GB | 1.36 GB | 73.2 ms | 20.2 ms | 0.931005 ✗ | 40.69 / 23.54 px |

The KV cache is FP16 in all three: NVFP4 KV needs `sm100f` (datacenter Blackwell) and Thor is
sm110, and FP8 KV is a separate strategy (`s2`). The vision tower stays BF16 under `s1`;
quantizing it is `s3`/`s4` and is FP8-only, because the ViT MLP `intermediate_size` is 3420
and 3420 / 16 = 213.75 does not divide by the NVFP4 block size.

Weight quantization error, mean relative over 21 projections in layers 0/13/27:
FP8 **2.67 %**, NVFP4 **9.45 %**.

#### System 1 — NextDiT diffusion head + memory block

System 1 ships **BF16**. FP8 was measured rather than assumed, and the measurement is why it
does not ship: see below. Engine weights are BF16; the engine *interface* is fp32/int64, and
passing bf16 tensors trips an assertion in the wrapper rather than converting silently.

| Component | weights | ONNX | engine | latency | cosine vs PyTorch |
|---|---|---|---|---|---|
| memory block (DAv2 + MemoryEncoder + QFormer) | BF16 | 200 MB | **104 MB** | **2.04 ms** | **0.999981** |
| `traj_dit`, one diffusion step | BF16 | 134 MB | **72 MB** | **5.85 ms** | **0.999508** |
| full trajectory (10 steps x 32 samples x 32 waypoints) | BF16 | — | — | **61.8 ms** | **0.999670** |
| **System 1 total** (memory + trajectory) | BF16 | 334 MB | **176 MB** | **63.8 ms · 15.7 Hz** | — |
| PyTorch `generate_traj` baseline | BF16 | — | — | 175.4 ms · 5.7 Hz | — |

TensorRT gives System 1 a **2.75x** speedup at identical output. Latency is flat in
`num_sample_trajs` on the PyTorch side (175.4 / 175.6 / 173.1 ms at 32 / 4 / 1), so the head
is launch-bound rather than compute-bound — which is also why moving it to engines pays.

#### FP8 on System 1 — measured, and not recommended

Unlike System 2, System 1 goes `torch.onnx.export` -> `trtexec`, and TensorRT does FP8 only
through **explicit** quantization: the Q/DQ nodes must already be in the ONNX. So this needed
a ModelOpt PTQ pass (`quantize_system1.py`) calibrated on real tensors captured from a live
System 2 -> System 1 run (`dump_system1_calib.py`, 40 real `traj_dit` batches over 4 VLN
samples). Verified in the graph: 328 / 328 FP8 Q/DQ pairs in `traj_dit`, 160 / 160 in the
memory block.

Waypoint deviation is the number to read. Cosine says how *aligned* two trajectories are;
deviation says how far apart the robot would actually end up, in the trajectory's own units,
against a mean per-waypoint reach of **0.2052**.

| Config | traj_dit | memory | engines | trajectory | traj cosine | waypoint dev. mean / median / p95 |
|---|---|---|---|---|---|---|
| **BF16 (shipped)** | BF16 72 MB | BF16 104 MB | **176 MB** | 61.8 ms | **0.999670** | **0.0032 / 0.0012 / 0.0117** |
| mixed | **FP8 39 MB** | BF16 104 MB | 143 MB | 48.4 ms | 0.988032 ✗ | 0.0154 / 0.0050 / 0.0521 |
| all FP8 | **FP8 39 MB** | **FP8 75 MB** | **114 MB** | 49.6 ms | 0.980213 ✗ | 0.0198 / 0.0053 / 0.0793 |

FP8 works — 1.55x smaller, 20 % faster on the diffusion loop, both configs well clear of
garbage — and it is still the wrong trade here:

- **The size saving is irrelevant at the system level.** 62 MB off 9.16 GB of deployed
  weights is 0.7 %.
- **The latency saving is nearly as small.** 12 ms off a 709 ms planning step is 1.7 %,
  because System 2 dominates by an order of magnitude.
- **The fidelity cost is not small.** Mean waypoint deviation goes 0.0032 -> 0.0198, a **6x**
  increase, and p95 goes 0.0117 -> 0.0793, which is 39 % of a typical waypoint's reach.

Both FP8 configs fall below the 0.99 gate, and the split shows why: quantizing `traj_dit`
alone already costs 0.9997 -> 0.9880. Its per-step error is only 0.999508 -> 0.997811, but the
sampler runs 10 steps and each one feeds the next, so a small per-call error compounds. The
memory block adds the rest (0.999981 -> 0.990204) and buys **nothing** in latency (2.09 ms
against 2.04 ms) — its Conv2d stays unquantized anyway, since the legacy ONNX exporter cannot
infer a convolution kernel shape through Q/DQ.

**Keep System 1 in BF16.** Quantization effort belongs on System 2, which is 98 % of both the
weights and the latency. The scripts stay in the recipe so the measurement is reproducible
and so the finding can be re-checked on a different System-1 configuration.

#### Both systems, one planning step

System 2 latency here is the full multi-image VLN step (~9 images, ~1764 image tokens) from
12 held-out samples, **not** the synthetic `llm_bench` figures above — different measurement,
so do not mix the two columns.

| Configuration | System 2 quantization | System 2 | System 1 | total | vs PyTorch |
|---|---|---|---|---|---|
| all PyTorch | BF16 | 1631 ms | 175.4 ms | 1806 ms | 1.00x |
| TensorRT, unquantized | FP16 | 770 ms | 63.8 ms | 834 ms | 2.17x |
| **TensorRT, recommended** | **FP8 E4M3 W8A8** (System 1 stays BF16) | **646 ms** | **63.8 ms** | **710 ms** | **2.54x** |
| TensorRT, System 1 also FP8 | FP8 both systems | 646 ms | 51.6 ms | 698 ms | 2.59x ✗ |

The last row is why System 1 stays BF16: 1.7 % off the step for 6x the waypoint deviation.

Total on-device weights for the recommended configuration: **7.62 + 1.36 + 0.18 = 9.16 GB**,
against 15.7 GB for the unquantized TensorRT path.

**FP8 is the recommended scheme.** Against BF16 it is 1.86x smaller and 1.65x/1.79x faster,
holds the bridge at 0.9919, and the median waypoint error does not get worse — it improves
slightly (27.05 -> 22.51 px), which is within the spread of a 42-sample set and should be read
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

Upstream InternNav ships **no** TensorRT or ONNX export at all — nothing in `internnav/`
references `trtexec`, `tensorrt` or `torch.onnx`. The System-1 conversion here is entirely
this recipe's, ported from the source project.

Sizes, latency and fidelity are in the matrix above. The engine I/O, verified by execution:

| Engine | inputs | output |
|---|---|---|
| `traj_dit` | `x[64,32,384]` f32, `timestep[64]` i64, `z_latents[64,*,768]` f32 | `output[64,32,384]` f32 |
| memory block | `images[T,3,224,224]` f32 | `memory_tokens[1,32,768]` f32 |

The `64` is `2 x num_sample_trajs`: `generate_traj` runs classifier-free guidance, so the
conditioning is `[null, real]` and the latents are duplicated.

**What stays in PyTorch on the host**, by design rather than omission: `action_encoder` and
`action_decoder` (two 3x384 linears), `pos_encoding`, `cond_projector` (the System 2 bridge),
and the `FlowMatchEulerDiscreteScheduler` loop itself. These are tiny or control-flow heavy;
the two engines cover the compute.

**End-to-end parity: both engines reproduce PyTorch.**

| | cosine vs PyTorch |
|---|---|
| `memory_tokens` (DAv2 + MemoryEncoder + QFormer engine) | **0.999981** |
| `traj_dit`, one diffusion step on the reference's own tensors | **0.999508** |
| full trajectory, 32 samples x 32 waypoints x 3 (rel-L2 0.0258) | **0.999670** |

The check runs in **two stages across two environments**, because no single interpreter has
both halves: InternNav is written against transformers 4.x (under 5.x it fails first on
`config.hidden_size`, then on `apply_chunking_to_forward`, with no natural end), while the
TensorRT bindings ship for Python 3.12 only.

Splitting it removes the conflict entirely. `verify/dump_system1_reference.py` runs the
PyTorch reference where InternNav works and writes its inputs *and* outputs to a `.pt`;
`verify/compare_system1_engines.py` reads that file where TensorRT works. The comparison
stays exact because the **same tensors** cross the boundary — the engines are fed the
reference's own inputs rather than regenerated ones.

```bash
# stage A, transformers 4.51 environment (Python 3.10)
INTERNNAV_PATH=~/InternNav PYTHONPATH=~/InternNav \
python verify/dump_system1_reference.py --output_path work/system1_reference.pt

# stage B, TensorRT environment (Python 3.12)
python verify/compare_system1_engines.py \
    --reference_path work/system1_reference.pt --engine_dir work/onnx \
    --bench_iters 20        # optional: also time each engine and the whole trajectory
```

Two details are worth keeping, because both produce a confident wrong answer:

- **Normalize before the memory block.** `generate_traj` divides by `_resnet_mean/_resnet_std`
  before `rgb_model`; `MemBlock`, the module that was exported to ONNX, does not, so it
  expects the already-normalized tensor. Feeding raw pixels made `memory_tokens` disagree
  with what `generate_traj` used (0.315) while each half stayed internally consistent, which
  reads exactly like a broken engine.
- **Capture the reference's starting noise, and probe one step.** `generate_traj` draws its
  latents mid-function, so reseeding in stage B does not reproduce them, and a different draw
  gives a different-but-valid trajectory (cosine ~0.31). Stage A therefore dumps the actual
  noise, plus one real `traj_dit` call with its inputs and output. The single-step number is
  what separates a bad engine from a bad reimplementation of the sampler loop — here it read
  0.9995 while the trajectory still read 0.31, which localized the fault to the harness.

### A note on the older numbers

An earlier revision reported z_latents of 0.99974 / 0.99985 / 0.99559 / 0.647 for
PyTorch / FP16 / FP8 / NVFP4 against an FP32 reference on 12 samples. The matrix above
supersedes those: it uses a BF16 reference, 42 samples, and the corrected checkpoint loader.
The ordering is the same and the conclusion is unchanged — FP8 passes, NVFP4 does not.

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
        ├── dump_system1_calib.py         # real System 1 calibration tensors (needs InternNav)
        ├── quantize_system1.py           # System 1 FP8 PTQ -> ONNX -> engine (measured, not shipped)
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

System 1 needs InternNav and its own two interpreters, since no single environment has both
InternNav (transformers 4.x) and the TensorRT bindings (Python 3.12):

    make export-system1
    PYTHON_PT=/path/to/py310/bin/python PYTHON_TRT=/path/to/py312/bin/python \
        make verify-system1

`make help` lists every target. Each script is also runnable directly; see the per-path
READMEs in `quantize/` and `trt-edgellm/`.

## CLI Reference

    quantize/quantize.py [options]

      --model_path PATH           Repackaged System 2 checkpoint (required)
      --output_path PATH          Destination for the quantized checkpoint (required)
      --strategy {s1,s2,s3,s4}    s1 LLM · s2 +KV cache · s3 +ViT · s4 +ViT+KV (default: s1)
      --scheme NAME               Preset from configs/schemes.yaml (default: fp8_default)
      --calib {auto,text,multimodal,vln}   Calibration source (default: auto)
      --calib_data_root PATH      Root for VLN calibration scenes
      --num_calib_samples N       Calibration samples (default: 512; image paths cap at 128)
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
