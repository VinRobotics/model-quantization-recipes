#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Score a System-2 checkpoint on held-out VLN episodes, without a simulator.

The official InternVLA-N1 metrics (SR, SPL, NE, OS, nDTW) are all closed-loop and need
Habitat or InternUtopia plus MP3D scenes. Neither is practical on a Jetson, so this
measures the thing that quantization actually threatens and that *can* be measured
offline: whether System 2 still emits the same navigation decision.

Two metrics, both against the LeRobot ground truth:

* ``pixel_goal_l2`` -- Euclidean distance between the predicted and annotated waypoint,
  in pixels of the 384x384 prompt image. This is the primary number: the waypoint is what
  System 1 consumes, so an error here is an error in where the robot goes.
* ``action_accuracy`` -- agreement on the discrete action token (STOP / up / left / right).

Both are computed per checkpoint on identical samples, so two runs are directly
comparable. The interesting quantity is the *delta* between an unquantized and a
quantized checkpoint, not either absolute value.

Prompts are built through ``prompt_builder``, the same path the deployed agent uses, so
what is scored is the deployed prompt rather than a hand-rebuilt approximation.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_builder as pb  # noqa: E402
from load_quantized import load_for_eval  # noqa: E402

# From InternNav's dataset: idx2actions = {0: STOP, 1: up, 2: left, 3: right, 5: down}
IDX2ACTION = {0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}
# The agent runs two turns. Turn 1 shows the level camera plus history and the model
# replies with a discrete action -- or with the look-down token when the next thing it
# needs is a waypoint. The robot then tilts, and turn 2 appends the single pitched frame,
# where the coordinate is produced. That is why every goal.125cm_0deg entry in the data is
# the [-1,-1] sentinel: a level camera cannot see a point on the floor.
#
# Scoring has to follow the same two turns. Feeding a pitched frame through a turn-1
# prompt puts the model off-distribution and it answers with an action instead of a
# coordinate, which reads as a 0% parse rate and looks like a model failure when it is a
# harness failure.
LEVEL_CAMERA = "125cm_0deg"
DEFAULT_CAMERA = "125cm_30deg"
LOOKDOWN_TOKEN = "\u2193"


def parse_xy(text: str):
    """Pull the first two integers out of a model reply, matching InternNav's parser."""
    nums = re.findall(r"-?\d+", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def parse_action(text: str):
    for token in ("STOP", "↑", "←", "→", "↓"):
        if token in text:
            return token
    return None


def discover_samples(data_root: str, max_samples: int, seed: int = 0,
                     camera: str = DEFAULT_CAMERA) -> list[dict]:
    """Collect (images, instruction, GT waypoint, GT action) tuples from LeRobot episodes."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rgb_key = f"observation.images.rgb.{camera}"
    goal_col = f"goal.{camera}"
    samples: list[dict] = []
    for meta in sorted(glob.glob(os.path.join(data_root, "**", "meta", "episodes.jsonl"),
                                 recursive=True)):
        scene = os.path.dirname(os.path.dirname(meta))
        episodes = [json.loads(line) for line in open(meta)]
        for ep in episodes:
            idx = ep["episode_index"]
            length = ep["length"]
            if length < 2:
                continue
            table = None
            for parquet in glob.glob(os.path.join(scene, "data", "**",
                                                  f"episode_{idx:06d}.parquet"),
                                     recursive=True):
                table = pq.read_table(parquet)
                break
            if table is None or goal_col not in table.schema.names:
                continue

            goals = table.column(goal_col).to_pylist()
            actions = table.column("action").to_pylist()
            # goal is [-1, -1] on frames that carry no waypoint (a pure turn or stop).
            # Scoring L2 against that sentinel would be meaningless, so sample only from
            # frames that actually annotate one.
            usable = [i for i in range(1, min(length, len(goals)))
                      if goals[i] is not None and int(goals[i][0]) >= 0]
            if not usable:
                continue
            t = int(usable[rng.integers(0, len(usable))])

            history = np.unique(np.linspace(0, t - 1, pb.NUM_HISTORY, dtype=np.int32)).tolist()
            level_key = f"observation.images.rgb.{LEVEL_CAMERA}"
            frames = [os.path.join(scene, "videos", "chunk-000", level_key,
                                   f"episode_{idx:06d}_{i}.jpg") for i in history + [t]]
            lookdown = os.path.join(scene, "videos", "chunk-000", rgb_key,
                                    f"episode_{idx:06d}_{t}.jpg")
            if not all(os.path.isfile(p) for p in frames + [lookdown]):
                continue

            action = actions[t]
            samples.append({
                "episode": os.path.basename(scene),
                # prompt_builder's "episode_idx" is the step index within the episode --
                # it derives the history frames from linspace(0, episode_idx-1) -- not the
                # episode number. Passing the episode number yields one <image> placeholder
                # and a count mismatch against the history frames.
                "episode_idx": t,
                "episode_number": idx,
                "instruction": (ep.get("tasks") or [""])[0],
                "images": frames + [lookdown],
                "turn": 2,
                "assistant_turn1": LOOKDOWN_TOKEN,
                "gt_goal": goals[t],
                "gt_action": IDX2ACTION.get(int(action) if action is not None else -1),
            })
            if len(samples) >= max_samples:
                return samples
    return samples


def evaluate(model_path: str, samples: list[dict], max_new_tokens: int,
             device: str) -> dict:
    model, processor, algo = load_for_eval(model_path, device=device)
    # The prompt images are resized to 384x384, so predicted and GT pixel coordinates
    # live in the same frame and the L2 is directly interpretable.
    l2, n_parsed, n_action_ok, n_action_total = [], 0, 0, 0
    replies = []
    t0 = time.time()

    for i, sample in enumerate(samples, 1):
        inputs = pb.build_sample_inputs(sample, processor).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0].strip()
        replies.append(reply)

        xy = parse_xy(reply)
        if xy is not None and sample["gt_goal"] is not None:
            n_parsed += 1
            gt = sample["gt_goal"]
            l2.append(float(np.hypot(xy[0] - float(gt[0]), xy[1] - float(gt[1]))))

        # Only meaningful on turn 1. At turn 2 the model emits a coordinate by design,
        # so scoring it against a movement action would always read 0% and look like a
        # regression rather than the protocol working.
        if sample.get("turn", 1) == 1 and sample["gt_action"] is not None:
            n_action_total += 1
            if parse_action(reply) == sample["gt_action"]:
                n_action_ok += 1

        if i % 5 == 0 or i == len(samples):
            print(f"      [{i}/{len(samples)}] {time.time() - t0:.0f}s elapsed", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "model_path": model_path,
        "quant_algo": algo,
        "n_samples": len(samples),
        "pixel_goal_l2_mean": float(np.mean(l2)) if l2 else None,
        "pixel_goal_l2_median": float(np.median(l2)) if l2 else None,
        "pixel_goal_parse_rate": n_parsed / max(len(samples), 1),
        "action_accuracy": n_action_ok / n_action_total if n_action_total else None,
        "n_action_scored": n_action_total,
        "seconds": round(time.time() - t0, 1),
        "replies": replies,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, action="append",
                   help="Checkpoint to score. Repeat to compare several on identical samples.")
    p.add_argument("--data_root", required=True, help="Held-out LeRobot episodes")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera", default=DEFAULT_CAMERA,
                   help="LeRobot camera setting to score against. Must be a pitched\n"
                        "one; the level 125cm_0deg carries no pixel goal.")
    p.add_argument("--output_path", default=None, help="Write results JSON here")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    samples = discover_samples(args.data_root, args.num_samples, seed=args.seed,
                               camera=args.camera)
    if not samples:
        print(f"[ERROR] no usable samples under {args.data_root}")
        return 1
    print(f"Scoring {len(samples)} held-out samples from "
          f"{len({s['episode'] for s in samples})} scene(s)\n")

    results = []
    for path in args.model_path:
        print(f"=== {path}")
        res = evaluate(path, samples, args.max_new_tokens, args.device)
        results.append(res)
        print(f"    quant           : {res['quant_algo'] or 'none (bf16)'}")
        print(f"    pixel_goal_l2   : mean {res['pixel_goal_l2_mean']:.2f} px, "
              f"median {res['pixel_goal_l2_median']:.2f} px"
              if res["pixel_goal_l2_mean"] is not None else "    pixel_goal_l2   : n/a")
        print(f"    parse_rate      : {100 * res['pixel_goal_parse_rate']:.1f}%")
        if res["action_accuracy"] is not None and res["n_action_scored"]:
            print(f"    action_accuracy : {100 * res['action_accuracy']:.1f}% "
                  f"({res['n_action_scored']} scored)")
        print(f"    took            : {res['seconds']}s\n")

    if len(results) > 1:
        base = results[0]
        print("=== delta vs " + os.path.basename(base["model_path"]))
        for res in results[1:]:
            name = os.path.basename(res["model_path"])
            if base["pixel_goal_l2_mean"] and res["pixel_goal_l2_mean"]:
                d = res["pixel_goal_l2_mean"] - base["pixel_goal_l2_mean"]
                print(f"    {name}: pixel_goal_l2 {d:+.2f} px")
            if (base["action_accuracy"] is not None and res["action_accuracy"] is not None
                    and base["n_action_scored"]):
                d = 100 * (res["action_accuracy"] - base["action_accuracy"])
                print(f"    {name}: action_accuracy {d:+.1f} pp")
            agree = sum(a == b for a, b in zip(base["replies"], res["replies"]))
            print(f"    {name}: identical replies {agree}/{len(res['replies'])}")

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
