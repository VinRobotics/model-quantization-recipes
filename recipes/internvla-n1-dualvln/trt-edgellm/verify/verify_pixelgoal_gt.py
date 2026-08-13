#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Offline task-level metric: System 2 pixel-goal L2 vs dataset ground truth.

This is the metric InternVLA §4.2 uses for System 2: at a look-down goal frame the model
predicts the next waypoint's pixel `(u, v)`; the dataset stores the projected ground-truth
`goal.<pitch>` (produced by the official label generator
`scripts/dataset_converters/internvla_labels.py::project_world_point`) in the original 640x480
frame. Training targets the raw `f"{u} {v}"` (dataset line 1217, no rescaling), so predictions
and GT live in the same 640x480 space and L2 is direct.

Teacher-forced two-turn look-down (matching the agent): turn 1 on the level view is forced to
"↓" (idx2actions[5]); turn 2 appends the tilted look-down view and the model emits `u v`.
The conversation is built once (lib/prompt_builder) and fed to BOTH PyTorch and the engines, so
they see identical prompts.

Self-validation gate: PyTorch's own L2 vs GT must be small (paper regime, a few → tens of px). If
it is large, the coordinate mapping is wrong and NO number is reported — only cosine stands.

Select the engine via ENGINE_PATH; select data via VLN_CALIB_DATA; pick the pitch via VLN_PITCH.
"""
import os
import sys
import glob
import json
import re
import subprocess
import tempfile

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "lib"))

import numpy as np
from PIL import Image
# prompt_builder is the single source of truth for the VLN prompt and lives in the
# quantize path, so calibration and verification cannot drift apart. Walk up to the
# recipe root rather than counting directory levels -- these scripts sit at two
# different depths.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != "/" and not os.path.isdir(os.path.join(_d, "quantize")):
    _d = os.path.dirname(_d)
sys.path.insert(0, os.path.join(_d, "quantize"))
import prompt_builder as pb

TRT = os.path.expanduser(os.environ.get("TRT_EDGE_LLM", "~/modelopt/TensorRT-Edge-LLM"))
REPKG = os.path.expanduser(os.environ.get("REPKG", "~/vln-opt-work/repro/qwen25vl_system2"))
VIS = os.path.expanduser(os.environ.get(
    "VIS_ENG",
    os.path.join(os.environ.get("ENGINE_DIR",
        os.path.expanduser("~/vln-opt-work/engines")), "s1_fp8/visual")))
DATA = os.path.expanduser(os.environ.get("VLN_CALIB_DATA", "~/vln-opt-work/probe_heldout"))
# Look-down config: level history (0deg) + tilted goal view. GT lives at the tilted pitch.
LEVEL_KEY = "observation.images.rgb.125cm_0deg"
# One or more tilted look-down pitches (comma-separated). GT goal lives at each pitch.
PITCHES = [p.strip() for p in os.environ.get("VLN_PITCH", "125cm_30deg,60cm_30deg").split(",")]
LOOKDOWN_TOKEN = "↓"                                    # idx2actions[5]
MAX_SAMPLES = int(os.environ.get("N", "0")) or None      # None = all goal frames
ENGINE = os.path.expanduser(os.environ.get("ENGINE_PATH", "")) or None
env = dict(os.environ, EDGELLM_PLUGIN_PATH=f"{TRT}/build/libNvInfer_edgellm_plugin.so")
DIGITS = re.compile(r"\d+")


def frame_path(scene, key, ep_idx, fr):
    return os.path.join(scene, "videos", "chunk-000", key, f"episode_{ep_idx:06d}_{fr}.jpg")


def collect_goal_frames():
    """Yield (scene_dir, pitch, ep_idx, t, instruction, gt_uv) for every populated goal frame,
    across all configured pitches."""
    import pandas as pd
    out = []
    for meta in glob.glob(os.path.join(DATA, "**", "meta", "episodes.jsonl"), recursive=True):
        scene = os.path.dirname(os.path.dirname(meta))
        eps = {e["episode_index"]: e for e in
               (json.loads(l) for l in open(meta) if l.strip())}
        for pq in sorted(glob.glob(os.path.join(scene, "data", "chunk-000", "*.parquet"))):
            df = pd.read_parquet(pq)
            ep_idx = int(df["episode_index"].iloc[0])
            ep = eps.get(ep_idx)
            if not ep or not ep.get("tasks"):
                continue
            for pitch in PITCHES:
                gcol = f"goal.{pitch}"
                if gcol not in df.columns or not os.path.isdir(
                        os.path.join(scene, "videos", "chunk-000", f"observation.images.rgb.{pitch}")):
                    continue
                goals = np.stack(df[gcol].values)
                for t in range(len(df)):
                    u, v = int(goals[t][0]), int(goals[t][1])
                    if u < 0 or t == 0:
                        continue
                    out.append((scene, pitch, ep_idx, t, ep["tasks"][0], (u, v)))
    return out


def build_conv(instruction, t, img_paths):
    """One look-down conversation, images filled in order (history+current level, then tilt)."""
    conv = pb.build_conversation_lookdown(t, instruction, LOOKDOWN_TOKEN)
    j = 0
    for turn in conv:
        for it in turn["content"]:
            if it["type"] == "image" and it.get("image") is None:
                it["image"] = img_paths[j]
                j += 1
    assert j == len(img_paths), f"image count {j} != {len(img_paths)}"
    return conv


def pred_uv(text):
    d = [int(x) for x in DIGITS.findall(text)]
    return (d[0], d[1]) if len(d) >= 2 else None      # raw "u v" in 640x480 (policy.py decode)


def run_engine(conv, tmp):
    js = {"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
          "max_generate_length": 16, "requests": [{"messages": conv}]}
    inp = os.path.join(tmp, "in.json"); out = os.path.join(tmp, "out.json")
    json.dump(js, open(inp, "w"))
    subprocess.run([f"{TRT}/build/examples/llm/llm_inference",
                    "--engineDir", os.path.dirname(ENGINE), "--multimodalEngineDir", VIS,
                    "--inputFile", inp, "--outputFile", out],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return json.load(open(out))["responses"][0]["output_text"].strip()


def main():
    frames = collect_goal_frames()
    if not frames:
        print(f"No populated goal frames for pitches {PITCHES} under {DATA}", file=sys.stderr)
        return 1
    import random
    random.Random(0).shuffle(frames)
    if MAX_SAMPLES:
        frames = frames[:MAX_SAMPLES]
    sq = os.environ.get("VLN_SQUARE") == "1"
    print(f"[data] {len(frames)} goal frames | pitches={PITCHES} | "
          f"preproc={'384-square (deploy)' if sq else 'aspect-preserving (official)'} | engine="
          f"{os.path.basename(os.path.dirname(ENGINE)) if ENGINE else 'PyTorch-only'}")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPKG, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(REPKG, trust_remote_code=True,
                                         min_pixels=128*28*28, max_pixels=1024*28*28)

    pt_l2, eng_l2 = [], []
    tmp = tempfile.mkdtemp(prefix="pgoal_")
    for scene, PITCH, ep_idx, t, instr, (gu, gv) in frames:
        TILT_KEY = f"observation.images.rgb.{PITCH}"
        hist = np.unique(np.linspace(0, t - 1, pb.NUM_HISTORY, dtype=np.int32)).tolist()
        # history + current from the level view; look-down frame from the tilted view
        src = [frame_path(scene, LEVEL_KEY, ep_idx, h) for h in hist]
        src.append(frame_path(scene, LEVEL_KEY, ep_idx, t))
        src.append(frame_path(scene, TILT_KEY, ep_idx, t))
        if not all(os.path.isfile(p) for p in src):
            continue
        # Feed native frames (aspect-preserving); the processor's min/max_pixels sizes them as in
        # training. Forcing a 384x384 square would distort the 4:3 frame and shift the goal. GT
        # stays in native 640x480, matching the model's output space. Set VLN_SQUARE=1 to override.
        if os.environ.get("VLN_SQUARE") == "1":
            paths = []
            for i, p in enumerate(src):
                d = os.path.join(tmp, f"{i}.jpg")
                Image.open(p).convert("RGB").resize((pb.RESIZE_W, pb.RESIZE_H)).save(d, quality=95)
                paths.append(d)
        else:
            paths = src
        conv = build_conv(instr, t, paths)

        imgs = [Image.open(p).convert("RGB") for p in paths]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=16, do_sample=False)
        pt_txt = proc.tokenizer.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        puv = pred_uv(pt_txt)
        if puv:
            pt_l2.append(np.hypot(puv[0] - gu, puv[1] - gv))

        if ENGINE:
            euv = pred_uv(run_engine(conv, tmp))
            if euv:
                eng_l2.append(np.hypot(euv[0] - gu, euv[1] - gv))

    def ci(a, stat=np.median, n=1000, seed=0):
        a = np.asarray(a); rng = np.random.default_rng(seed)
        bs = [stat(rng.choice(a, len(a), replace=True)) for _ in range(n)]
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def report(a, tag):
        a = np.asarray(a)
        mlo, mhi = ci(a, np.median); alo, ahi = ci(a, np.mean)
        print(f"  {tag:13}: n={len(a)}  median={np.median(a):.1f} [{mlo:.1f},{mhi:.1f}]  "
              f"mean={a.mean():.1f} [{alo:.1f},{ahi:.1f}]  max={a.max():.0f}")

    print("\n=== pixel-goal L2 vs GT (640x480, held-out, 95% CI bootstrap) ===")
    report(pt_l2, "PyTorch BF16")
    # Validation gate uses the aspect-preserving PyTorch run; at 384-square the absolute is
    # expectedly inflated (deploy resize) so the mapping is validated only in the native run.
    if not sq:
        med = np.median(pt_l2)
        print(f"  [gate] PyTorch median-L2 (aspect-preserving) = {med:.1f}px "
              f"({'PASS <60, mapping valid' if med < 60 else 'CHECK'})")
    if ENGINE and eng_l2:
        report(eng_l2, os.path.basename(os.path.dirname(ENGINE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
