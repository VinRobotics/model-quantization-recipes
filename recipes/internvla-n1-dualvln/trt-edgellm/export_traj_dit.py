#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Step 4 - export the System 1 traj_dit (NextDiT) core to ONNX with a dynamic
z_latents length and build the BF16 engine (trtexec).
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/usr/lib/python3.12/dist-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import torch
from traj_dit_loader import load_traj_dit

OUT = os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_traj_dit_async.onnx")
ENG = os.path.join(os.environ.get("WORK_DIR",os.path.expanduser("~/vln-opt-work")), "onnx/system1_traj_dit_bf16.engine")
ZLEN = int(os.environ.get("ZLEN", "36"))  # observed z_latents length for the async path


def main():
    dev="cuda"; N,WP,DIM=32,32,384
    print(f"[1/3] Load traj_dit (FP32) | z_latents seq (async, dynamic, opt={ZLEN})")
    dit,zdim = load_traj_dit(); dit=dit.to(torch.float32).eval()
    x=torch.randn(2*N,WP,DIM,dtype=torch.float32,device=dev)
    ts=torch.ones(2*N,dtype=torch.int64,device=dev)
    z=torch.randn(2*N,ZLEN,zdim,dtype=torch.float32,device=dev)
    class W(torch.nn.Module):
        def __init__(s,d): super().__init__(); s.d=d
        def forward(s,x,timestep,z_latents): return s.d(x=x,timestep=timestep,z_latents=z_latents)
    w=W(dit).eval()
    with torch.no_grad(): ref=w(x,ts,z)
    print(f"      forward OK -> {tuple(ref.shape)}")
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    # dynamic: batch (dim0) + z_latents seq (dim1)
    dyn={"x":{0:"batch"},"timestep":{0:"batch"},"z_latents":{0:"batch",1:"zlen"},"output":{0:"batch"}}
    print(f"[2/3] Export ONNX (dynamo=False, opset 19) → {OUT}")
    with torch.inference_mode():
        torch.onnx.export(w,(x,ts,z),OUT,input_names=["x","timestep","z_latents"],
            output_names=["output"],opset_version=19,do_constant_folding=True,
            export_params=True,dynamic_axes=dyn,dynamo=False)
    print(f"      {os.path.getsize(OUT)/1e6:.1f} MB")
    B=2*N
    cmd=["/usr/src/tensorrt/bin/trtexec",f"--onnx={OUT}",f"--saveEngine={ENG}","--bf16",
         f"--minShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x4x{zdim}",
         f"--optShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x{ZLEN}x{zdim}",
         f"--maxShapes=x:{B}x{WP}x{DIM},timestep:{B},z_latents:{B}x64x{zdim}"]
    print(f"[3/3] Build engine BF16 (z dynamic 4..64, opt {ZLEN})\n      {' '.join(cmd)}")
    r=subprocess.run(cmd,capture_output=True,text=True)
    print("  "+ "\n  ".join(l for l in r.stdout.splitlines() if "successfully" in l.lower() or "PASSED" in l or "FAILED" in l)[-400:])
    print("  engine:", ENG, os.path.getsize(ENG)/1e6 if os.path.exists(ENG) else "MISSING", "MB")
    return 0 if os.path.exists(ENG) else 1


if __name__=="__main__":
    sys.exit(main())
