#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Verify System 2 accuracy against the documented reference agent
(InternVLAN1AsyncAgent) run verbatim on sample data: branch decision + pixel goal,
compared frame-by-frame for both PyTorch and the FP8 TensorRT engine.
"""
import os
import sys
import json
import subprocess
import glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from engine_runner import ENGINE  # noqa: E402

ACTIVE = os.environ.get("INTERNNAV_PATH", os.path.expanduser("~/InternNav"))
CKPT = os.path.join(ACTIVE, "checkpoints/InternVLA-N1-DualVLN")
TRT = os.environ.get("TRT_EDGE_LLM", os.path.expanduser("~/TensorRT-Edge-LLM"))
ENG_LLM = os.path.dirname(ENGINE)
ENG_VIS = os.path.join(os.environ.get("WORK_DIR", os.path.expanduser("~/vln-opt-work")), "engines/system2_visual")
SCRATCH = os.path.expanduser(os.environ.get('VLN_OPT_OUT', '~/vln-opt-work/out'))
# camera_intrinsic matches the demo (inference_only_demo cell 15)
INTR = np.array([[386.5, 0, 328.9, 0], [0, 386.5, 244, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class Args:
    device = "cuda:0"
    model_path = CKPT
    model_path_original = CKPT
    resize_w = 384
    resize_h = 384
    num_history = 8
    plan_step_gap = 4


def main():
    if ACTIVE not in sys.path:
        sys.path.insert(0, ACTIVE)
    from internvla_compat import apply_all
    apply_all(need_system1=True, allow_missing_depth=True)
    import importlib.util
    _s = importlib.util.spec_from_file_location(
        "iar", os.path.join(ACTIVE, "internnav/agent/internvla_n1_agent_realworld.py"))
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    Agent = _m.InternVLAN1AsyncAgent

    print("[1/4] Load the documented reference agent")
    agent = Agent(Args())
    dbg = os.path.join(SCRATCH, "verb_dbg")
    os.makedirs(dbg, exist_ok=True)
    IMG_DIR = os.path.join(SCRATCH, "verb_imgs")
    os.makedirs(IMG_DIR, exist_ok=True)

    cap = {}
    orig = agent.model.generate

    def hooked(*a, **kw):
        cap["fired"] = True
        cap["input_ids"] = kw.get("input_ids")
        return orig(*a, **kw)
    agent.model.generate = hooked

    scenes = sorted(g for g in glob.glob(os.path.join(ACTIVE, "assets/realworld_sample_data*"))
                    if os.path.isdir(g))
    samples = []
    for scene in scenes:
        sname = os.path.basename(scene)
        instr = open(os.path.join(scene, "instruction.txt")).read().strip()
        # documented demo order: sorted debug_raw_*.jpg (look_down frames interleaved)
        rgb_paths = sorted(glob.glob(os.path.join(scene, "debug_raw_*.jpg")))
        print(f"[2/4] {sname} | instr={instr!r} | {len(rgb_paths)} frames (docs loop)")
        agent.reset()
        agent.save_dir = dbg
        for p in rgb_paths:
            look_down = ('look_down' in p)
            rgb = np.asarray(Image.open(p).convert('RGB'))
            depth = 10 * np.ones((rgb.shape[0], rgb.shape[1]), np.float32)   # docs: fill depth
            pose = np.eye(4)
            cap.clear()
            try:
                with torch.no_grad():
                    agent.step(rgb, depth, pose, instr, intrinsic=INTR, look_down=look_down)
            except Exception as e:
                print(f"    {os.path.basename(p)}: step err {type(e).__name__}: {e}")
                continue
            if not cap.get("fired"):
                continue  # S2 did not run on this frame (buffer) - skip
            ref_out = agent.llm_output.strip()
            msgs, ii = [], 0
            for turn in agent.conversation_history:
                content = []
                for it in turn["content"]:
                    if it["type"] == "image":
                        fp = os.path.join(IMG_DIR, f"s{len(samples)}_{ii}.png")
                        it["image"].save(fp)
                        content.append({"type": "image", "image": fp})
                        ii += 1
                    else:
                        content.append({"type": "text", "text": it["text"]})
                msgs.append({"role": turn["role"], "content": content})
            samples.append(dict(scene=sname, img=os.path.basename(p), look_down=look_down,
                                ref=ref_out, S=int(cap["input_ids"].shape[1]) if cap.get(
                                    "input_ids") is not None else -1,
                                messages=msgs))
            print(f"    {os.path.basename(p):32} look_down={look_down} S={samples[-1]['S']} ref={ref_out!r}")

    in_json = os.path.join(SCRATCH, "verb_in.json")
    out_json = os.path.join(SCRATCH, "verb_out.json")
    json.dump({"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_generate_length": 128,
               "requests": [{"messages": s["messages"]} for s in samples]}, open(in_json, "w"))
    del agent
    torch.cuda.empty_cache()

    print(f"[3/4] Engine {os.path.basename(os.path.dirname(ENGINE))} through llm_inference ({len(samples)} req)")
    env = dict(os.environ, EDGELLM_PLUGIN_PATH=f"{TRT}/build/libNvInfer_edgellm_plugin.so")
    r = subprocess.run([f"{TRT}/build/examples/llm/llm_inference", "--engineDir", ENG_LLM,
                        "--multimodalEngineDir", ENG_VIS, "--inputFile", in_json, "--outputFile", out_json],
                       cwd=TRT, env=env, capture_output=True, text=True)
    resp = json.load(open(out_json)).get("responses", []) if os.path.exists(out_json) else []
    print(f"      exit={r.returncode}, {len(resp)} responses")

    print("[4/4] Compare engine against the reference verbatim\n" + "=" * 66)
    import re

    def xy(t):
        d = [int(c) for c in re.findall(r"\d+", t)]
        return (d[0], d[1]) if len(d) >= 2 else None
    ex = br = 0
    l2 = []
    for i, s in enumerate(samples):
        fp = resp[i].get("output_text", "").strip() if i < len(resp) else "<none>"
        s["eng"] = fp
        rb = "coord" if any(c.isdigit() for c in s["ref"]) else "action"
        fb = "coord" if any(c.isdigit() for c in fp) else "action"
        ex += s["ref"] == fp
        br += rb == fb
        a, b = xy(s["ref"]), xy(fp)
        if a and b:
            l2.append(((a[0] - b[0])**2 + (a[1] - b[1])**2)**.5)
        print(f"  [{'OK' if s['ref']==fp else 'DIFF':4}] {s['img']:32} ref={s['ref']!r:18} eng={fp[:40]!r}")
    n = len(samples)
    print("\n" + "=" * 66)
    print(f"  exact-match  : {ex}/{n} = {ex/n*100:.1f}%")
    print(f"  branch-agree : {br}/{n} = {br/n*100:.1f}%")
    if l2:
        print(
            f"  coord-L2 px  : mean={sum(l2)/len(l2):.1f} median={sorted(l2)[len(l2)//2]:.1f} max={max(l2):.0f} (n={len(l2)})")  # noqa: E501
    tag = os.path.basename(os.path.dirname(ENGINE)).replace("edgellm_engines_", "")
    json.dump(samples, open(os.path.join(os.environ.get("VLN_OPT_OUT", os.path.expanduser("~/vln-opt-work/out")), f"verbatim_{tag}.json"), "w"),  # noqa: E501
              indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
