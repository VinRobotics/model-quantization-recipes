#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Verify System 1: wire the traj_dit + memory TensorRT engines into generate_traj and
compare the resulting trajectory and latency against the PyTorch base (same noise seed).
"""
import os, sys, time
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "lib"))
sys.path.append("/usr/lib/python3.12/dist-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"lib"))
import numpy as np, torch
from PIL import Image
from memblock import MemBlock

ACTIVE=os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT=os.path.join(ACTIVE,"checkpoints/InternVLA-N1-DualVLN")
IMG=os.path.expanduser("~/modelopt/TensorRT-Edge-LLM/examples/multimodal/pics/giant_panda.jpeg")
TRAJDIT=os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_traj_dit_bf16.engine")
MEM=os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_memory_bf16.engine")
SEED=12345


def out_of(o, key):
    if isinstance(o, dict): return o.get(key, next(iter(o.values())))
    return o[0] if isinstance(o,(list,tuple)) else o


def main():
    dev="cuda"
    try: torch.backends.mha.set_fastpath_enabled(False)
    except Exception: pass
    if ACTIVE not in sys.path: sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor
    from trt_torch import Engine
    print("[1/4] Load model + engines")
    cfg=InternVLAN1ModelConfig.from_pretrained(CKPT)
    model=InternVLAN1ForCausalLM.from_pretrained(CKPT,config=cfg,torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",low_cpu_mem_usage=True).to(dev).eval()
    m=model.get_model()
    mem_eng=Engine(MEM); dit_eng=Engine(TRAJDIT)
    NQ=model.get_n_query()
    z=torch.randn(1,NQ,cfg.hidden_size,device=dev,dtype=torch.bfloat16)
    a=np.array(Image.open(IMG).convert("RGB").resize((224,224)))/255.0
    tt=torch.from_numpy(a).float(); images_dp=torch.stack([tt,tt]).unsqueeze(0).to(dev)
    rmean=model._resnet_mean; rstd=model._resnet_std

    def base_gen():
        torch.manual_seed(SEED); np.random.seed(SEED)
        with torch.no_grad():
            return model.generate_traj(z,images_dp,num_sample_trajs=32,num_inference_steps=10).float().cpu()

    def trt_gen(steps=10, ns=32, guidance_scale=1.0, predict_step_nums=32):
        """Replicate generate_traj (async) with the memory + traj_dit engines."""
        torch.manual_seed(SEED); np.random.seed(SEED)
        dtype=z.dtype
        with torch.no_grad():
            traj_latents = m.cond_projector(z)                                  # [1,4,768]
            # --- memory block through the engine ---
            xdp = images_dp.permute(0,1,4,2,3)
            xdp = ((xdp - rmean)/rstd).flatten(0,1)                              # [2,3,224,224]
            memory_tokens = out_of(mem_eng(images=xdp.float().contiguous()), "memory_tokens").to(dtype)  # [1,32,768]
            hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)     # [1,36,768]
            hs_null = torch.zeros_like(hidden_states)
            hs_input = torch.cat([hs_null, hidden_states], 0)                    # [2,36,768]
            bs = traj_latents.shape[0]
            latents = randn_tensor((bs*ns, predict_step_nums, 3), generator=None, device=dev, dtype=dtype)
            sch = FlowMatchEulerDiscreteScheduler()
            sigmas = np.linspace(1.0, 1/steps, steps)
            sch.set_timesteps(steps, sigmas=sigmas)
            hs_input = hs_input.repeat_interleave(ns, dim=0)                     # [2*ns,36,768]
            dit_eng.set_runtime_tensor_shape("z_latents", tuple(hs_input.shape))
            for t in sch.timesteps:
                lf = m.action_encoder(latents)
                pos_ids = torch.arange(lf.shape[1]).reshape(1,-1).repeat(bs*ns,1).to(dev)
                lf = lf + m.pos_encoding(pos_ids)
                lmi = lf.repeat(2,1,1)
                if hasattr(sch,"scale_model_input"): lmi = sch.scale_model_input(lmi, t)
                tt_ = t.unsqueeze(0).expand(lmi.shape[0]).to(dev, torch.long)
                np_ = out_of(dit_eng(x=lmi.float().contiguous(), timestep=tt_.to(torch.int64).contiguous(),
                                     z_latents=hs_input.float().contiguous()), "output").to(dtype)
                np_ = m.action_decoder(np_)
                unc, cnd = np_.chunk(2)
                np_ = unc + guidance_scale*(cnd - unc)
                latents = sch.step(np_, t, latents).prev_sample
            return latents.float().cpu()

    def lat(fn,warm=2,n=5):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        return sum(ts)/len(ts)*1000

    print("[2/4] Base PyTorch generate_traj")
    tref = base_gen(); pt_ms = lat(base_gen)
    print(f"    {tuple(tref.shape)} | {pt_ms:.1f}ms = {1000/pt_ms:.1f}Hz")
    print("[3/4] Full-TRT S1 (memory+traj_dit engines)")
    ttrt = trt_gen(); trt_ms = lat(trt_gen)
    print(f"    {tuple(ttrt.shape)} | {trt_ms:.1f}ms = {1000/trt_ms:.1f}Hz")
    print("[4/4] Compare\n"+"="*56)
    d=(tref-ttrt).norm(dim=-1); endp=(tref[:,-1]-ttrt[:,-1]).norm(dim=-1)
    cos=torch.nn.functional.cosine_similarity(tref.flatten(),ttrt.flatten(),dim=0).item()
    print(f"  Parity: per-wp L2 mean={d.mean():.4f} max={d.max():.4f} | endpoint={endp.mean():.4f} | cos={cos:.5f}")
    print(f"  Latency S1: PyTorch {pt_ms:.1f}ms ({1000/pt_ms:.1f}Hz) → full-TRT {trt_ms:.1f}ms ({1000/trt_ms:.1f}Hz) = {pt_ms/trt_ms:.2f}x")
    return 0


if __name__=="__main__":
    sys.exit(main())
