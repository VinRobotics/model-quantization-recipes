#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Verify System 2 numeric fidelity: z_latents cosine of the FP8 LLM engine vs the
PyTorch reference, using the real latent-query bridge path.
"""
import os, sys, json
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); sys.path.insert(0, _R)
import torch
from PIL import Image
from engine_runner import (REPKG, ENGINE, CKPT, build_mrope_table, run_engine,
                                 cos, per_tok_cos)

TRAJ_TOKEN_INDEX = 151667
IMAGE_TOKEN_INDEX = 151655
N_QUERY = 4
IMAGE = os.path.expanduser(os.environ.get(
    "IMAGE_PATH", "~/modelopt/TensorRT-Edge-LLM/examples/multimodal/pics/giant_panda.jpeg"))


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
    print(f"[1/6] Load repackage + processor | engine={os.path.basename(ENGINE)}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPKG, torch_dtype=torch.bfloat16, # flash-attn is not available on Jetson; sdpa is the supported path and
        # is what the deployed agent uses too.
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"),
        low_cpu_mem_usage=True).to(dev).eval()
    proc = AutoProcessor.from_pretrained(REPKG, trust_remote_code=True,
                                         min_pixels=128*28*28, max_pixels=1024*28*28)
    inner = model.model                       # Qwen2_5_VLModel (takes inputs_embeds)
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    final_norm = lm.norm

    print("[2/6] latent_queries + cond_projector from the InternVLA checkpoint")
    lq = load_ckpt_tensor("model.latent_queries")["model.latent_queries"].to(dev).to(torch.bfloat16)
    cpw = load_ckpt_tensor("model.cond_projector")
    cp = {k.replace("model.cond_projector.", ""): v.float().to(dev) for k, v in cpw.items()}
    print(f"      latent_queries {tuple(lq.shape)} (n_query={lq.shape[1]})  cond keys {sorted(cp.keys())}")
    assert lq.shape[1] == N_QUERY

    print("[3/6] Build image + instruction input, append 4 TRAJ tokens (as generate_latents)")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": IMAGE},
        {"type": "text", "text": "Go straight then stop at the green plant."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[Image.open(IMAGE).convert("RGB")], return_tensors="pt").to(dev)
    input_ids = enc["input_ids"]
    grid = enc["image_grid_thw"]
    # embed + scatter image + append latent_queries
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)                      # [1,S,3584]
        image_embeds = visual_embeds(model, enc["pixel_values"], grid)
    image_idx = (input_ids == IMAGE_TOKEN_INDEX)
    text_embeds[image_idx] = image_embeds.to(text_embeds.dtype)[: image_idx.sum(), :]
    inputs_embeds = torch.cat([text_embeds, lq.repeat(text_embeds.shape[0], 1, 1)], dim=1)
    ids_traj = torch.cat(
        [input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY], device=dev)], dim=1)
    S = inputs_embeds.shape[1]
    position_ids, _ = rope_index(model, ids_traj, grid, IMAGE_TOKEN_INDEX)
    print(f"      seq_len={S} (+{N_QUERY} traj)  grid={grid.tolist()}  pos {tuple(position_ids.shape)}")

    print("[4/6] Reference forward (inner LM); capture pre/post norm at the last 4 positions")
    cap = {}
    h = final_norm.register_forward_hook(lambda m, i, o: cap.update(pre=i[0].detach(), post=o.detach()))
    with torch.no_grad():
        inner(inputs_embeds=inputs_embeds, position_ids=position_ids,
              output_hidden_states=True, return_dict=True)
    h.remove()
    ref_pre = cap["pre"][:, -N_QUERY:, :].float()
    ref_post = cap["post"][:, -N_QUERY:, :].float()

    print("[5/6] Engine forward (3D mRoPE); take hidden[-4:] pre-norm -> host norm")
    rope = build_mrope_table(position_ids, dev)
    _, eng_hs = run_engine(inputs_embeds.to(torch.float16), rope)
    eng_pre = eng_hs[:, -N_QUERY:, :].float()
    with torch.no_grad():
        eng_post = final_norm(eng_pre.to(torch.bfloat16)).float()

    print("[6/6] Compare z_latents (System1 traj_dit input)\n" + "=" * 60)
    def cond_project(x):
        x = torch.nn.functional.linear(x, cp["0.weight"], cp.get("0.bias"))
        x = torch.nn.functional.gelu(x)
        return torch.nn.functional.linear(x, cp["2.weight"], cp.get("2.bias"))
    z_ref, z_eng = cond_project(ref_post), cond_project(eng_post)
    ptc = torch.nn.functional.cosine_similarity(eng_pre[0], ref_pre[0], dim=-1)
    print("  per-query PRE cosine:", " ".join(f"{ptc[i]:.4f}" for i in range(N_QUERY)))
    print(f"  hidden PRE-norm  cosine = {per_tok_cos(eng_pre, ref_pre):.6f}")
    print(f"  hidden POST-norm cosine = {per_tok_cos(eng_post, ref_post):.6f}")
    zc = per_tok_cos(z_eng, z_ref)
    zc_flat = cos(z_eng, z_ref)
    l2 = (z_eng - z_ref).norm() / z_ref.norm()
    print(f"  z_latents({z_eng.shape[-1]}) per-query cosine = {zc:.6f} | flat = {zc_flat:.6f} "
          f"| rel-L2 = {l2:.4f}")
    ok = zc > 0.99
    print("\n" + (f"FAITHFUL e2e bridge PASS - z_latents match (cosine {zc:.4f})"
                  if ok else f"⚠️ z_latents cosine {zc:.4f} < 0.99"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
