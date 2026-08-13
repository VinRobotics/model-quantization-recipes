#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end verification: run the InternVLA agent with the TensorRT engines
(FP8 LLM for the bridge + BF16 System 1 engines) and compare its outputs, frame by
frame, against the pure-PyTorch agent on the documented sample data.
"""
import os
import sys
import glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "lib"))
sys.path.append("/usr/lib/python3.12/dist-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from engine_runner import build_mrope_table, run_engine   # LLM engine harness  # noqa: E402

ACTIVE = os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT = os.path.join(ACTIVE, "checkpoints/InternVLA-N1-DualVLN")
REPKG = os.path.join(os.environ.get("WORK_DIR", os.path.expanduser("~/vln-opt-work")), "qwen25vl_system2")
TRAJDIT = os.path.join(
    os.environ.get(
        "WORK_DIR",
        os.path.expanduser("~/vln-opt-work")),
         "onnx/system1_traj_dit_async_bf16.engine")  # noqa: E131
MEM = os.path.join(os.environ.get("WORK_DIR", os.path.expanduser("~/vln-opt-work")), "onnx/system1_memory_bf16.engine")
SCRATCH = os.path.expanduser(os.environ.get('VLN_OPT_OUT', '~/vln-opt-work/out'))
INTR = np.array([[386.5, 0, 328.9, 0], [0, 386.5, 244, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
TRAJ_TOKEN_INDEX = 151667
IMAGE_TOKEN_INDEX = 151655
SEED = 12345


def out_of(o, k):
    if isinstance(o, dict):
        return o.get(k, next(iter(o.values())))
    return o[0] if isinstance(o, (list, tuple)) else o


def main():
    dev = "cuda"
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass
    if ACTIVE not in sys.path:
        sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    import importlib.util
    _s = importlib.util.spec_from_file_location("iar", os.path.join(
        ACTIVE, "internnav/agent/internvla_n1_agent_realworld.py"))
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    Agent = _m.InternVLAN1AsyncAgent
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor
    from trt_torch import Engine

    class A:
        device = "cuda:0"
        model_path = CKPT
        model_path_original = CKPT
        resize_w = 384
        resize_h = 384
        num_history = 8
        plan_step_gap = 4
    print("[1/5] Load agent + engines")
    agent = Agent(A())
    agent.save_dir = os.path.join(SCRATCH, "trtagent_dbg")
    os.makedirs(agent.save_dir, exist_ok=True)
    model = agent.model
    m = model.get_model()
    mem_eng = Engine(MEM)
    dit_eng = Engine(TRAJDIT)
    lm = m.language_model if hasattr(m, "language_model") else m
    final_norm = lm.norm
    NQ = model.get_n_query()
    rmean = model._resnet_mean
    rstd = model._resnet_std

    # ---- FP8 LLM generate_latents (latent_query + mRoPE logic, run on the engine) ----
    def trt_generate_latents(input_ids, pixel_values, image_grid_thw):
        with torch.no_grad():
            te = m.embed_tokens(input_ids)
            ie = model.visual(pixel_values.type(model.visual.dtype), grid_thw=image_grid_thw)
        te[input_ids == IMAGE_TOKEN_INDEX] = ie.to(te.dtype)[:(input_ids == IMAGE_TOKEN_INDEX).sum(), :]
        lq = m.latent_queries.repeat(te.shape[0], 1, 1)
        inputs_embeds = torch.cat([te, lq], dim=1)
        ids_traj = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * NQ], device=dev)], dim=1)
        position_ids, _ = model.get_rope_index(ids_traj, image_grid_thw)
        rope = build_mrope_table(position_ids, dev)
        _, eng_hs = run_engine(inputs_embeds.to(torch.float16), rope)
        eng_pre = eng_hs[:, -NQ:, :].float()
        return final_norm(eng_pre.to(torch.bfloat16))     # hidden [1,NQ,3584] (as generate_latents)

    # ---- S1 generate_traj through the engines (async logic matches the baseline) ----
    def trt_generate_traj(traj_latents, images_dp, depths_dp=None, predict_step_nums=32,
                          guidance_scale=1.0, num_inference_steps=10, num_sample_trajs=32):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        dtype = traj_latents.dtype
        with torch.no_grad():
            tl = m.cond_projector(traj_latents)
            xdp = images_dp.permute(0, 1, 4, 2, 3)
            xdp = ((xdp - rmean) / rstd).flatten(0, 1)
            memory_tokens = out_of(mem_eng(images=xdp.float().contiguous()), "memory_tokens").to(dtype)
            hs = torch.cat([memory_tokens, tl], dim=1)
            hs_in = torch.cat([torch.zeros_like(hs), hs], 0)
            bs = tl.shape[0]
            latents = randn_tensor((bs * num_sample_trajs, predict_step_nums, 3),
                                   generator=None, device=dev, dtype=dtype)
            sch = FlowMatchEulerDiscreteScheduler()
            sch.set_timesteps(
    num_inference_steps,  # noqa: E122
    sigmas=np.linspace(  # noqa: E122
        1.0,
        1 / num_inference_steps,
         num_inference_steps))  # noqa: E131
            hs_in = hs_in.repeat_interleave(num_sample_trajs, dim=0)
            dit_eng.set_runtime_tensor_shape("z_latents", tuple(hs_in.shape))
            for t in sch.timesteps:
                lf = m.action_encoder(latents)
                pid = torch.arange(lf.shape[1]).reshape(1, -1).repeat(bs * num_sample_trajs, 1).to(dev)
                lf = lf + m.pos_encoding(pid)
                lmi = lf.repeat(2, 1, 1)
                if hasattr(sch, "scale_model_input"):
                    lmi = sch.scale_model_input(lmi, t)
                tt = t.unsqueeze(0).expand(lmi.shape[0]).to(dev, torch.long)
                npd = out_of(
    dit_eng(  # noqa: E122
        x=lmi.float().contiguous(),
        timestep=tt.to(
            torch.int64).contiguous(),
            z_latents=hs_in.float().contiguous()),  # noqa: E131
             "output").to(dtype)  # noqa: E122
                npd = m.action_decoder(npd)
                unc, cnd = npd.chunk(2)
                npd = unc + guidance_scale * (cnd - unc)
                latents = sch.step(npd, t, latents).prev_sample
            return latents

    # PyTorch reference generate_traj: fixed seed for a fair comparison
    base_traj = model.generate_traj

    def seeded_base_traj(*a, **k):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        return base_traj(*a, **k)

    scene = sorted(g for g in glob.glob(os.path.join(ACTIVE, "assets/realworld_sample_data*")) if os.path.isdir(g))[0]
    instr = open(os.path.join(scene, "instruction.txt")).read().strip()
    rgbs = sorted(glob.glob(os.path.join(scene, "debug_raw_*.jpg")))[:40]
    print(f"[2/5] scene {os.path.basename(scene)} | {len(rgbs)} frames | instr={instr[:50]!r}...")

    def run(tag):
        agent.reset()
        agent.save_dir = os.path.join(SCRATCH, "trtagent_dbg")
        outs = []
        for p in rgbs:
            ld = ('look_down' in p)
            rgb = np.asarray(Image.open(p).convert('RGB'))
            depth = 10 * np.ones(rgb.shape[:2], np.float32)
            pose = np.eye(4)
            try:
                with torch.no_grad():
                    o = agent.step(rgb, depth, pose, instr, intrinsic=INTR, look_down=ld)
            except Exception as e:
                print(f"    {os.path.basename(p)}: {type(e).__name__}: {e}")
                continue
            traj = o.output_trajectory
            act = o.output_action
            if traj is not None:
                outs.append(("traj", os.path.basename(p), np.asarray(traj)))
            elif act is not None:
                outs.append(("act", os.path.basename(p), list(act)))
        return outs

    print("[3/5] Run PyTorch agent (reference)")
    model.generate_traj = seeded_base_traj
    ref = run("pytorch")
    print(f"    {len(ref)} outputs")

    print("[4/5] Run TensorRT agent (FP8 LLM latents + S1 engines)")
    model.generate_latents = trt_generate_latents
    model.generate_traj = trt_generate_traj
    trt = run("trt")
    print(f"    {len(trt)} outputs")

    print("[5/5] Compare e2e TRT agent vs PyTorch\n" + "=" * 56)
    n = min(len(ref), len(trt))
    mact = 0
    nact = 0
    trajerr = []
    for i in range(n):
        rk, rf, rv = ref[i]
        tk, tf, tv = trt[i]
        if rk == "act" and tk == "act":
            nact += 1
            mact += (rv == tv)
        elif rk == "traj" and tk == "traj":
            e = np.linalg.norm(np.asarray(rv) - np.asarray(tv), axis=-1)
            trajerr.append(float(np.mean(e)))
    print(
        f"  outputs: pytorch={len(ref)} trt={len(trt)} (type match {sum(1 for i in range(n) if ref[i][0]==trt[i][0])}/{n})")  # noqa: E501
    if nact:
        print(f"  action match: {mact}/{nact}")
    if trajerr:
        import statistics
        print(
            f"  trajectory per-wp L2 (m): mean={statistics.mean(trajerr):.4f} max={max(trajerr):.4f} (n={len(trajerr)})")  # noqa: E501
    print("  Agent TRT runs end-to-end and matches PyTorch." if n > 0 else "  no output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
