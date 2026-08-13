#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Verify System 2 z_latents fidelity on the REAL VLN input distribution.

Same z_latents bridge comparison as ``verify_system2_latents.py`` (FP8 engine vs
PyTorch reference), but instead of a single caption image it drives the model with
genuine InternData-N1 VLN-CE steps (multi-image history + navigation instruction +
4 appended TRAJ tokens), averaged over several samples. This is the probe that
actually reflects deployment, so it is the fair test of whether VLN-distribution
FP8 calibration helps — a single-image caption probe is out-of-distribution.

Select the engine with ENGINE_PATH; select the calibration/probe data with
VLN_CALIB_DATA. Uses a fixed seed so every engine sees identical inputs.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "build", "quantize"))

import torch

from engine_runner import (REPKG, ENGINE, CKPT, build_mrope_table, run_engine,
                           cos, per_tok_cos)

TRAJ_TOKEN_INDEX = 151667
IMAGE_TOKEN_INDEX = 151655
N_QUERY = 4
N_SAMPLES = int(os.environ.get("VLN_PROBE_SAMPLES", "12"))
DATA = os.path.expanduser(os.environ.get(
    "VLN_CALIB_DATA",
    "~/vln-opt-work/calib_scenes"))   # output of build/00_fetch_calib_scenes.sh; override via env


def load_ckpt_tensor(prefix):
    """Read bridge tensors, preferring the small bridge.safetensors next to the checkpoint.

    repackage_system2.py sets latent_queries and cond_projector aside in a ~25 MB
    bridge.safetensors precisely so this check does not have to reopen the 16 GB original.
    Falling back to a full sharded checkpoint keeps the script usable against one.
    """
    from safetensors import safe_open

    bridge = os.path.join(CKPT, "bridge.safetensors")
    if os.path.isfile(bridge):
        with safe_open(bridge, framework="pt") as f:
            return {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}

    index = os.path.join(CKPT, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise FileNotFoundError(
            f"neither bridge.safetensors nor model.safetensors.index.json under {CKPT}; "
            f"point INTERNVLA_CKPT at a repackaged checkpoint or the original")
    idx = json.load(open(index))["weight_map"]
    out = {}
    for k in idx:
        if k.startswith(prefix):
            with safe_open(os.path.join(CKPT, idx[k]), framework="pt") as f:
                out[k] = f.get_tensor(k)
    return out


def rope_index(model, input_ids, image_grid_thw, image_token_id):
    """Call get_rope_index across the two transformers signatures.

    transformers 4.x took (input_ids, image_grid_thw). 5.x inserted a required
    mm_token_type_ids argument -- 0 text, 1 image, 2 video -- and moved the grids to
    keywords. Passing the 4.x form to a 5.x model does not raise on arity; it fails later
    inside with 'NoneType is not an iterator', which is a confusing way to learn this.
    """
    import inspect

    fn = model_attr(model, "get_rope_index")
    params = inspect.signature(fn).parameters
    if "mm_token_type_ids" in params:
        mm = (input_ids == image_token_id).to(torch.int32)
        return fn(input_ids, mm, image_grid_thw=image_grid_thw)
    return fn(input_ids, image_grid_thw)


def model_attr(model, name):
    """Fetch an attribute or bound method from the model or its inner model.

    transformers 5.x moved several Qwen2.5-VL helpers (get_rope_index, visual, ...) from
    the ForConditionalGeneration wrapper down onto the inner Qwen2_5_VLModel. Probing both
    keeps this working on either layout instead of pinning a transformers version.
    """
    for owner in (model, getattr(model, "model", None)):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    raise AttributeError(f"neither the model nor its inner model has {name!r}")


def get_visual(model):
    """Return the vision tower across transformers layouts.

    Older versions expose it as ``model.visual``; newer ones nest it under
    ``model.model.visual``. Probing beats pinning a transformers version here.
    """
    for owner in (model, getattr(model, "model", None)):
        vis = getattr(owner, "visual", None) if owner is not None else None
        if vis is not None:
            return vis
    raise AttributeError("no vision tower found on this model "
                         "(looked at .visual and .model.visual)")


def visual_embeds(model, pixel_values, grid_thw):
    """Return merged image embeddings at LLM hidden width, across transformers layouts.

    In transformers 5.x get_image_features returns a BaseModelOutputWithPooling whose
    ``pooler_output`` holds the *merged* embeddings (split per image), while
    ``last_hidden_state`` is the pre-merger tensor at vision width -- 1280 here against the
    LLM's 3584. Reading the wrong field fails loudly on the shape, but only after the
    vision tower has already run, so unwrap explicitly.
    """
    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values.type(get_visual(model).dtype), grid_thw)
        feats = getattr(out, "pooler_output", out)
    else:
        vis = get_visual(model)
        out = vis(pixel_values.type(vis.dtype), grid_thw=grid_thw)
        feats = getattr(out, "last_hidden_state", out)
    if isinstance(feats, (list, tuple)):
        feats = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)
    return feats.reshape(-1, feats.shape[-1])


def main():
    dev = "cuda"
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from calibration import vln_calib_dataloader

    print(f"[1/4] Load repackage + processor | engine={os.path.basename(ENGINE)}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPKG, torch_dtype=torch.bfloat16, # flash-attn is not available on Jetson; sdpa is the supported path and
        # is what the deployed agent uses too.
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"),
        low_cpu_mem_usage=True).to(dev).eval()
    proc = AutoProcessor.from_pretrained(REPKG, trust_remote_code=True,
                                         min_pixels=128 * 28 * 28, max_pixels=1024 * 28 * 28)
    inner = model.model
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    final_norm = lm.norm

    lq = load_ckpt_tensor("model.latent_queries")["model.latent_queries"].to(dev).to(torch.bfloat16)
    cpw = load_ckpt_tensor("model.cond_projector")
    cp = {k.replace("model.cond_projector.", ""): v.float().to(dev) for k, v in cpw.items()}
    assert lq.shape[1] == N_QUERY

    def cond_project(x):
        x = torch.nn.functional.linear(x, cp["0.weight"], cp.get("0.bias"))
        x = torch.nn.functional.gelu(x)
        return torch.nn.functional.linear(x, cp["2.weight"], cp.get("2.bias"))

    print(f"[2/4] Build {N_SAMPLES} real VLN inputs (multi-image + instruction) | data={DATA}")
    batches = vln_calib_dataloader(proc, data_root=DATA, num_samples=N_SAMPLES, seed=0)

    print("[3/4] For each: reference LM forward vs engine forward at the 4 TRAJ positions")
    hid_pre, hid_post, zc_list, zflat_list = [], [], [], []
    for bi, enc in enumerate(batches):
        input_ids = enc["input_ids"].to(dev)
        grid = enc["image_grid_thw"].to(dev)
        pixel_values = enc["pixel_values"].to(dev)
        with torch.no_grad():
            text_embeds = model.get_input_embeddings()(input_ids)
            image_embeds = model.visual(pixel_values.type(model.visual.dtype), grid_thw=grid)
        image_idx = (input_ids == IMAGE_TOKEN_INDEX)
        text_embeds[image_idx] = image_embeds.to(text_embeds.dtype)[: image_idx.sum(), :]
        inputs_embeds = torch.cat([text_embeds, lq.repeat(text_embeds.shape[0], 1, 1)], dim=1)
        ids_traj = torch.cat(
            [input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY], device=dev)], dim=1)
        position_ids, _ = rope_index(model, ids_traj, grid, IMAGE_TOKEN_INDEX)

        cap = {}
        h = final_norm.register_forward_hook(
            lambda m, i, o: cap.update(pre=i[0].detach(), post=o.detach()))
        with torch.no_grad():
            inner(inputs_embeds=inputs_embeds, position_ids=position_ids,
                  output_hidden_states=True, return_dict=True)
        h.remove()
        ref_pre = cap["pre"][:, -N_QUERY:, :].float()
        ref_post = cap["post"][:, -N_QUERY:, :].float()

        rope = build_mrope_table(position_ids, dev)
        _, eng_hs = run_engine(inputs_embeds.to(torch.float16), rope)
        eng_pre = eng_hs[:, -N_QUERY:, :].float()
        with torch.no_grad():
            eng_post = final_norm(eng_pre.to(torch.bfloat16)).float()

        z_ref, z_eng = cond_project(ref_post), cond_project(eng_post)
        hp = per_tok_cos(eng_pre, ref_pre)
        hpo = per_tok_cos(eng_post, ref_post)
        zc = per_tok_cos(z_eng, z_ref)
        zf = cos(z_eng, z_ref)
        hid_pre.append(hp); hid_post.append(hpo); zc_list.append(zc); zflat_list.append(zf)
        print(f"  #{bi:2d} imgs={grid.shape[0]:2d} seq={input_ids.shape[1]:4d} "
              f"| hidPRE={hp:.5f} hidPOST={hpo:.5f} z={zc:.5f}")

    import statistics as st
    n = len(zc_list)
    print("\n[4/4] Mean over VLN inputs\n" + "=" * 60)
    print(f"  samples          : {n}")
    print(f"  hidden PRE-norm  : {st.mean(hid_pre):.6f}  (min {min(hid_pre):.6f})")
    print(f"  hidden POST-norm : {st.mean(hid_post):.6f}  (min {min(hid_post):.6f})")
    print(f"  z_latents perq   : {st.mean(zc_list):.6f}  (min {min(zc_list):.6f})")
    print(f"  z_latents flat   : {st.mean(zflat_list):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
