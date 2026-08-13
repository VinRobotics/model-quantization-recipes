#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Step 5 - export the System 1 memory block (DINOv2 rgb_model + memory_encoder +
rgb_resampler) to ONNX and build the BF16 engine.
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/usr/lib/python3.12/dist-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, torch
from PIL import Image

ACTIVE=os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT=os.path.join(ACTIVE,"checkpoints/InternVLA-N1-DualVLN")
IMG=os.path.expanduser("~/modelopt/TensorRT-Edge-LLM/examples/multimodal/pics/giant_panda.jpeg")
ONNX=os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_memory_bf16.onnx")
ENG=os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_memory_bf16.engine")


# MemBlock lives in memblock.py; defining it twice is how the two copies drift.
from memblock import MemBlock


def main():
    dev="cuda"
    # disable fused MHA fastpath (_transformer_encoder_layer_fwd is not ONNX-exportable)
    try: torch.backends.mha.set_fastpath_enabled(False)
    except Exception as e: print("  (mha fastpath toggle:", e, ")")
    if ACTIVE not in sys.path: sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    from internnav.model.basemodel.internvla_n1.internvla_n1 import (
        InternVLAN1ForCausalLM, InternVLAN1ModelConfig)
    print("[1/6] Load full model")
    cfg=InternVLAN1ModelConfig.from_pretrained(CKPT)
    model=InternVLAN1ForCausalLM.from_pretrained(CKPT,config=cfg,torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",low_cpu_mem_usage=True).to(dev).eval()
    m=model.get_model()

    print("[2/6] Build input (matching the reference normalization)")
    a=np.array(Image.open(IMG).convert("RGB").resize((224,224)))/255.0
    tt=torch.from_numpy(a).float(); images_dp=torch.stack([tt,tt]).unsqueeze(0).to(dev)  # [1,2,224,224,3]
    dtype=torch.bfloat16
    rmean = model._resnet_mean if hasattr(model,"_resnet_mean") else m._resnet_mean
    rstd  = model._resnet_std  if hasattr(model,"_resnet_std")  else m._resnet_std
    x = images_dp.permute(0,1,4,2,3)                              # [1,2,3,224,224]
    x = (x - rmean)/rstd
    x = x.flatten(0,1).to(dtype)                                  # [2,3,224,224]
    print(f"      input {tuple(x.shape)}")

    print("[3/6] Ref base (PyTorch BF16) memory_tokens + latency")
    block = MemBlock(m.rgb_model, m.memory_encoder, m.rgb_resampler).eval()
    with torch.no_grad(): ref = block(x).float()
    print(f"      memory_tokens {tuple(ref.shape)}")
    def lat(fn,warm=3,n=10):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); ts=[]
        for _ in range(n):
            torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        return sum(ts)/len(ts)*1000
    with torch.no_grad(): pt_ms=lat(lambda: block(x))

    print("[4/6] Export ONNX (FP32 to avoid mixed precision; build BF16 later)")
    block_fp32 = MemBlock(m.rgb_model.float(), m.memory_encoder.float(), m.rgb_resampler.float()).eval()
    xf = x.float()
    os.makedirs(os.path.dirname(ONNX),exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(block_fp32,(xf,),ONNX,input_names=["images"],output_names=["memory_tokens"],
            opset_version=19,do_constant_folding=True,export_params=True,dynamo=False)
    print(f"      {os.path.getsize(ONNX)/1e6:.1f} MB")

    print("[5/6] Build BF16 engine (fixed shape 2x3x224x224)")
    cmd=["/usr/src/tensorrt/bin/trtexec",f"--onnx={ONNX}",f"--saveEngine={ENG}","--bf16"]  # static shape
    r=subprocess.run(cmd,capture_output=True,text=True)
    print(f"      build {'OK' if os.path.exists(ENG) else 'FAIL'} | {os.path.getsize(ENG)/1e6 if os.path.exists(ENG) else 0:.1f} MB")
    if not os.path.exists(ENG):
        print("  STDERR:", r.stderr[-600:]); return 1

    print("[6/6] Parity + latency (TRT vs base)")
    # The parity check needs the TensorRT Python bindings, which JetPack ships for
    # Python 3.12 only -- while this export needs transformers 4.51, which lives in the
    # 3.10 environment. The engine is already built and valid at this point, so a missing
    # binding must not turn a successful build into a failure. Run
    # verify/verify_system1.py under the 3.12 environment to check parity.
    try:
        from trt_torch import Engine
    except ImportError as exc:
        print(f"\n[skip] parity check unavailable in this interpreter: {exc}")
        print("       The engine was built successfully. To check it, run")
        print("       verify/verify_system1.py under the TensorRT (Python 3.12) environment.")
        return 0
    eng=Engine(ENG)
    out=eng(images=x.float().contiguous())
    out=(out.get("memory_tokens") if isinstance(out,dict) else out).float()
    cos=torch.nn.functional.cosine_similarity(ref.flatten(),out.flatten(),dim=0).item()
    trt_ms=lat(lambda: eng(images=x.float().contiguous()))
    print(f"      parity cos={cos:.5f} rel-L2={(ref-out).norm()/ref.norm():.4f}")
    print(f"      latency PyTorch {pt_ms:.2f}ms → TRT {trt_ms:.2f}ms = {pt_ms/trt_ms:.2f}x")
    return 0


if __name__=="__main__":
    sys.exit(main())
