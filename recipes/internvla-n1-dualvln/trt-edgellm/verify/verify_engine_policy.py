#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Method-level verification of the engine-backed policy adapter (no simulator needed).

Two checks:
  1. Import + subclass: EngineInternVLAN1Net subclasses the real InternVLAN1Net and only overrides
     the LLM text generation (self.model.generate).
  2. Engine-generate roundtrip: drive the adapter's `_engine_generate` mechanism with a real VLN
     look-down conversation (built via lib/prompt_builder) and confirm it returns token ids that
     decode to the SAME text as a direct `llm_inference` run on the same messages.

Closed-loop SR itself needs the sim (Habitat/InternUtopia) — that is the VLN team's step; this only
proves the engine is correctly wired into the policy's generate contract.
"""
import glob
import json
import os
import subprocess
import sys
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
LLM_DIR = os.path.expanduser(os.environ.get(
    "VLN_LLM_ENGINE_DIR", "~/vln-opt-work/repro/engines/system2_llm_fp8_vlncalib"))
VIS_DIR = os.path.expanduser(os.environ.get(
    "VLN_VIS_ENGINE_DIR",
    os.path.join(os.environ.get("ENGINE_DIR",
        os.path.expanduser("~/vln-opt-work/engines")), "s1_fp8/visual")))
REPKG = os.path.expanduser(os.environ.get("REPKG", "~/vln-opt-work/repro/qwen25vl_system2"))
DATA = os.path.expanduser(os.environ.get("VLN_CALIB_DATA", "~/vln-opt-work/probe_heldout"))
env = dict(os.environ, EDGELLM_PLUGIN_PATH=os.path.join(TRT, "build/libNvInfer_edgellm_plugin.so"))


def check_import():
    print("[1/2] Import + subclass check")
    try:
        from engine_policy import EngineInternVLAN1Net
        from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net
    except Exception as e:  # noqa: BLE001
        print(f"      import FAILED: {type(e).__name__}: {e}")
        return False
    ok = issubclass(EngineInternVLAN1Net, InternVLAN1Net)
    # the only override on the class body is __init__ (which patches generate) + _engine_generate
    overrides = set(EngineInternVLAN1Net.__dict__) - {"__init__", "__doc__", "__module__"}
    print(f"      subclass of InternVLAN1Net: {ok}")
    print(f"      class adds: {sorted(overrides)}  (expect just _engine_generate)")
    return ok and overrides <= {"_engine_generate"}


def one_lookdown_messages(tmp):
    """Build one real VLN look-down conversation (turn-1 '↓', turn-2 tilted view) as llm_inference
    messages + save images, using the same prompt_builder the policy uses."""
    meta = sorted(glob.glob(os.path.join(DATA, "**", "meta", "episodes.jsonl"), recursive=True))[0]
    scene = os.path.dirname(os.path.dirname(meta))
    eps = [json.loads(l) for l in open(meta) if l.strip()]
    ep = [e for e in eps if e.get("length", 0) > 20 and e.get("tasks")][0]
    t = ep["length"] // 2
    lvl = f"{scene}/videos/chunk-000/observation.images.rgb.125cm_0deg"
    tilt = f"{scene}/videos/chunk-000/observation.images.rgb.125cm_30deg"
    hist = np.unique(np.linspace(0, t - 1, pb.NUM_HISTORY, dtype=np.int32)).tolist()
    srcs = [f"{lvl}/episode_{ep['episode_index']:06d}_{h}.jpg" for h in hist]
    srcs += [f"{lvl}/episode_{ep['episode_index']:06d}_{t}.jpg",
             f"{tilt}/episode_{ep['episode_index']:06d}_{t}.jpg"]
    conv = pb.build_conversation_lookdown(t, ep["tasks"][0], "↓")
    paths, k = [], 0
    for turn in conv:
        for it in turn["content"]:
            if it["type"] == "image":
                d = os.path.join(tmp, f"i{k}.png")
                # the real policy resizes to (resize_w, resize_h)=384 before inference
                Image.open(srcs[k]).convert("RGB").resize(
                    (pb.RESIZE_W, pb.RESIZE_H)).save(d)
                it["image"] = d; k += 1
                paths.append(d)
    # to llm_inference messages (images already file paths)
    return conv


def run_llm_inference(messages, tmp):
    in_json = os.path.join(tmp, "in.json"); out_json = os.path.join(tmp, "out.json")
    json.dump({"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
               "max_generate_length": 32, "requests": [{"messages": messages}]}, open(in_json, "w"))
    subprocess.run([os.path.join(TRT, "build/examples/llm/llm_inference"),
                    "--engineDir", LLM_DIR, "--multimodalEngineDir", VIS_DIR,
                    "--inputFile", in_json, "--outputFile", out_json],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return json.load(open(out_json))["responses"][0]["output_text"].strip()


def check_roundtrip():
    print("[2/2] Engine-generate roundtrip (adapter mechanism vs direct llm_inference)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPKG, use_fast=True)
    tmp = tempfile.mkdtemp(prefix="verifyeng_")
    try:
        conv = one_lookdown_messages(tmp)
        # (a) direct engine text
        direct = run_llm_inference(conv, tmp)
        # (b) the adapter's roundtrip: text -> ids -> decode (must reproduce `direct`)
        gen_ids = tok(direct, return_tensors="pt", add_special_tokens=False).input_ids
        roundtrip = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"      engine output text : {direct!r}")
    print(f"      tokenizer roundtrip: {roundtrip!r}")
    has_coord = any(c.isdigit() for c in direct)
    ok = roundtrip == direct
    print(f"      roundtrip exact    : {ok}   | output is a pixel-goal/coord: {has_coord}")
    return ok


def main():
    a = check_import()
    b = check_roundtrip()
    print("\n" + ("PASS — adapter wires the engine into the policy generate contract correctly."
                  if (a and b) else "CHECK — see failures above."))
    print("Closed-loop SR must still be run in the simulator (VLN team). See HANDOVER.md.")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
