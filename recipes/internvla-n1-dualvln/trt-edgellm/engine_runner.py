#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Shared TensorRT engine runner + mRoPE table builder for the System 2 LLM engine.

Drives the FP8 LLM engine directly from Python: builds the 3D mRoPE cos/sin table from the
reference `position_ids` (per-token, merged by mrope_section (16,24,24), layout [cos64|sin64]),
binds torch GPU buffers to the TensorRT context, and returns (logits, hidden_states) for a
single prefill. Used by the verification scripts. Select an engine via the ENGINE_PATH env var.
"""
import os, sys, ctypes, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# TensorRT ships with JetPack outside the venv.
sys.path.append(os.environ.get("SYSTEM_SITE", "/usr/lib/python3.12/dist-packages"))
import numpy as np, tensorrt as trt, torch
from PIL import Image

def _env(name, default):
    return os.path.expanduser(os.environ.get(name, default))


WORK_DIR = _env("WORK_DIR", "~/vln-opt-work")
REPKG = _env("REPKG_CKPT", os.path.join(WORK_DIR, "qwen25vl_system2"))
ENGINE = _env("ENGINE_PATH", os.path.join(WORK_DIR, "engines/s1_fp8/llm/llm.engine"))
TRT_EDGELLM_DIR = _env("TRT_EDGELLM_DIR", "~/modelopt/TensorRT-Edge-LLM")
PLUGIN = _env("EDGELLM_PLUGIN_PATH",
              os.path.join(TRT_EDGELLM_DIR, "build/libNvInfer_edgellm_plugin.so"))
# The bridge tensors are emitted next to the repackaged checkpoint by
# repackage_system2.py, so the 16 GB original is not needed here.
CKPT = _env("INTERNVLA_CKPT", REPKG)
IMAGE = os.path.expanduser(os.environ.get(
    "IMAGE_PATH", os.path.join(TRT_EDGELLM_DIR,
                 "examples/multimodal/pics/giant_panda.jpeg")))

# Qwen2.5-VL-7B / InternVLA-N1 System 2 geometry. Inlined rather than imported: this
# repository has no shared-library convention, each recipe stands alone.
THETA = 1_000_000.0
HEAD_DIM = 128
N_LAYERS = 28
N_KV = 4
HIDDEN = 3584
ROPE_MAXPOS = 4096          # must match --maxKVCacheCapacity at build time
MROPE_SECTION = [16, 24, 24]  # (T, H, W)
TRT2TORCH = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
             trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
             trt.DataType.BF16: torch.bfloat16}
_ENG = {}


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def per_tok_cos(a, b):
    a, b = a[0].float(), b[0].float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def build_mrope_table(position_ids, device):
    """position_ids [3,1,S] → rope table [1, ROPE_MAXPOS, 128] (rows 0..S-1 are per-token
    mRoPE cos/sin merged by mrope_section; layout [cos64|sin64])."""
    S = position_ids.shape[-1]
    half = HEAD_DIM // 2  # 64
    zid = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = THETA ** (-2.0 * zid / HEAD_DIM)              # [64]
    # axis index per freq band: [0]*16 + [1]*24 + [2]*24
    axis = torch.cat([torch.full((MROPE_SECTION[i],), i, device=device) for i in range(3)])  # [64]
    pos = position_ids[:, 0, :].float()                      # [3,S]
    # per token s, freq j: angle = pos[axis[j], s] * inv_freq[j]
    pos_sel = pos[axis]                                      # [64,S]
    ang = pos_sel.T[:, :] * inv_freq[None, :]                # [S,64]
    c, s = torch.cos(ang), torch.sin(ang)                    # [S,64]
    table = torch.zeros(1, ROPE_MAXPOS, HEAD_DIM, dtype=torch.float32, device=device)
    table[0, :S, :half] = c
    table[0, :S, half:] = s
    return table


def load_engine():
    if "eng" not in _ENG:
        ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)
        lg = trt.Logger(trt.Logger.ERROR); trt.init_libnvinfer_plugins(lg, "")
        rt = trt.Runtime(lg)
        with open(ENGINE, "rb") as f:
            _ENG["eng"] = rt.deserialize_cuda_engine(f.read()); _ENG["rt"] = rt
    return _ENG["eng"]


def run_engine(embeds_half, rope_table):
    S = embeds_half.shape[1]; dev = embeds_half.device
    eng = load_engine(); ctx = eng.create_execution_context()
    ctx.set_optimization_profile_async(0, torch.cuda.current_stream().cuda_stream)
    context_lengths = torch.tensor([S], dtype=torch.int32, device=dev)
    kvcache_start = torch.zeros(1, dtype=torch.int32, device=dev)
    last_token = torch.tensor([[S - 1]], dtype=torch.int64, device=dev)
    kv_cap = ROPE_MAXPOS
    kv_cache = [torch.zeros(1, 2, N_KV, kv_cap, HEAD_DIM, dtype=torch.float16, device=dev)
                for _ in range(N_LAYERS)]
    feed = {
        "inputs_embeds": (embeds_half.contiguous(), None),
        "rope_rotary_cos_sin": (rope_table.contiguous(), None),
        "context_lengths": (context_lengths, None),
        "kvcache_start_index": (kvcache_start, (0,)),
        "last_token_ids": (last_token, None),
    }
    for i in range(N_LAYERS):
        feed[f"past_key_values_{i}"] = (kv_cache[i], (1, 2, N_KV, kv_cap, HEAD_DIM))
    for name, (t, shp) in feed.items():
        ctx.set_input_shape(name, shp if shp else tuple(t.shape))
        ctx.set_tensor_address(name, t.data_ptr())
    outs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) != trt.TensorIOMode.OUTPUT:
            continue
        if n.startswith("present_key_values_"):
            li = int(n.rsplit("_", 1)[1])
            ctx.set_tensor_address(n, kv_cache[li].data_ptr()); continue
        shp = tuple(int(d) for d in ctx.get_tensor_shape(n))
        shp = tuple(S if d < 0 else d for d in shp)
        t = torch.empty(shp, dtype=TRT2TORCH[eng.get_tensor_dtype(n)], device=dev)
        outs[n] = t; ctx.set_tensor_address(n, t.data_ptr())
    outs["_kv"] = kv_cache
    ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(); assert ok
    return outs["logits"], outs["hidden_states"]


def main():
    dev = "cuda"; torch.manual_seed(0)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"[1/5] Load repackage (transformers) | engine={os.path.basename(ENGINE)}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPKG, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True).to(dev).eval()
    proc = AutoProcessor.from_pretrained(REPKG, trust_remote_code=True,
                                         min_pixels=128*28*28, max_pixels=1024*28*28)
    backbone = model.model.language_model if hasattr(model.model, "language_model") else model.model
    final_norm = backbone.norm

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE},
        {"type": "text", "text": "Please describe the image."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[Image.open(IMAGE).convert("RGB")],
               return_tensors="pt").to(dev)
    S = inp["input_ids"].shape[1]
    print(f"      seq_len={S}  (image grid_thw={inp['image_grid_thw'].tolist()})")

    # hook inner LM to capture the reference inputs_embeds + position_ids
    cap = {}
    def pre_hook(mod, args, kwargs):
        cap["inputs_embeds"] = kwargs.get("inputs_embeds")
        cap["position_ids"] = kwargs.get("position_ids")
    h1 = backbone.register_forward_pre_hook(pre_hook, with_kwargs=True)
    def norm_hook(mod, i, o):
        cap["pre"] = i[0].detach(); cap["post"] = o.detach()
    h2 = final_norm.register_forward_hook(norm_hook)

    print("[2/5] Reference forward (WITH image)")
    with torch.no_grad():
        ref_out = model(**inp, use_cache=False)
    h1.remove(); h2.remove()
    embeds = cap["inputs_embeds"]; pos = cap["position_ids"]
    if embeds is None:
        # fallback: build inputs_embeds externally
        raise RuntimeError("inputs_embeds not captured - wrong hook target")
    print(f"      captured inputs_embeds {tuple(embeds.shape)} position_ids {tuple(pos.shape)}")
    print(f"      position_ids axis-range T[{pos[0].min()}..{pos[0].max()}] "
          f"H[{pos[1].min()}..{pos[1].max()}] W[{pos[2].min()}..{pos[2].max()}]")
    ref_pre = cap["pre"].float(); ref_post = cap["post"].float()
    ref_logits_last = ref_out.logits[:, -1, :].float()

    print("[3/5] Build merged 3D mRoPE + run engine")
    rope = build_mrope_table(pos, dev)
    eng_logits, eng_hs = run_engine(embeds.to(torch.float16), rope)
    eng_hs = eng_hs.float()
    with torch.no_grad():
        eng_post = final_norm(eng_hs.to(torch.bfloat16)).float()

    print("[4/5] Compare hidden_states / logits\n" + "=" * 60)
    a, b = eng_hs[0].float(), ref_pre[0].float()
    ptc = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print("  per-pos PRE cosine[:8]:", " ".join(f"{ptc[i]:.3f}" for i in range(min(S, 8))),
          "... last:", f"{ptc[-1]:.3f}")
    print(f"  hidden PRE-norm  per-token cosine = {per_tok_cos(eng_hs, ref_pre):.6f}")
    print(f"  hidden POST-norm per-token cosine = {per_tok_cos(eng_post, ref_post):.6f}")
    ea, ra = eng_logits[0, 0].argmax().item(), ref_logits_last[0].argmax().item()
    print(f"  logits last cosine = {cos(eng_logits[0,0], ref_logits_last[0]):.6f} "
          f"| argmax eng={ea!r} ref={ra!r} match={ea==ra}")

    print("[5/5] z_latents (the original cond_projector) engine vs reference")
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    cp = {}
    for k in idx:
        if k.startswith("model.cond_projector"):
            with safe_open(os.path.join(CKPT, idx[k]), framework="pt") as f:
                cp[k.replace("model.cond_projector.", "")] = f.get_tensor(k).float().to(dev)
    def cond_project(x):
        x = torch.nn.functional.linear(x, cp["0.weight"], cp.get("0.bias"))
        x = torch.nn.functional.gelu(x)
        return torch.nn.functional.linear(x, cp["2.weight"], cp.get("2.bias"))
    z_eng, z_ref = cond_project(eng_post), cond_project(ref_post)
    zc = per_tok_cos(z_eng, z_ref)
    print(f"  z_latents({z_eng.shape[-1]}) per-token cosine = {zc:.6f}")
    print("\n" + ("✅ WITH-IMAGE numeric PASS (z_latents ≥ 0.99)" if zc > 0.99
                  else f"z_latents cosine {zc:.4f} < 0.99 - check mRoPE/vision"))
    return 0 if zc > 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
