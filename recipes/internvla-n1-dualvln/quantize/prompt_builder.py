#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Prompt / conversation builders for InternVLA-N1 System 2.

Reconstructs the exact multi-image chat prompt that ``InternVLAN1Net.s2_step`` builds, for both
the normal turn and the look-down (turn 2) follow-up. Kept as a single source of truth so the
verification and calibration scripts never diverge from the agent's real prompt format.

Read-only: only a processor/tokenizer is loaded; no model weights, no writes to any source tree.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor

# Eval config values (InternNav h1_internvla_n1_async_cfg).
RESIZE_W = 384
RESIZE_H = 384
NUM_HISTORY = 8

# Special token ids (internvla_n1.py).
IMAGE_TOKEN_INDEX = 151655
TRAJ_TOKEN_INDEX = 151667

# Base prompt template (internvla_n1_policy.py), verbatim.
PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)
CONJUNCTION = "you can see "
DEFAULT_IMAGE_TOKEN = "<image>"

SAMPLE_INSTRUCTION = (
    "walk out of the bathroom and turn left, then walk down the hallway and "
    "stop in front of the second door on your right"
)

# Two processor construction variants: the deploy default vs the calibration override.
PROCESSOR_VARIANTS: dict[str, dict[str, Any]] = {
    "deploy": {},
    "calib": {"min_pixels": 128 * 28 * 28, "max_pixels": 2048 * 32 * 32},
}


def split_and_clean(text: str) -> list[str]:
    """Split around '<image>' and drop empty parts.

    Mirrors ``internnav.model.utils.vln_utils.split_and_clean`` (re-implemented so this module
    runs without the full internnav dependency tree).
    """
    import re

    parts = re.split(r"(<image>)", text)
    return [p for p in (s.strip() if s != DEFAULT_IMAGE_TOKEN else s for s in parts) if p]


def build_conversation(episode_idx: int, instruction: str) -> tuple[list[dict], int]:
    """Build the normal-turn prompt (look_down=False), matching ``internvla_n1_policy.py``.

    Returns ``(conversation, n_images)``.
    """
    sources_value = PROMPT_TEMPLATE.replace("<instruction>.", instruction)

    if episode_idx == 0:
        history_id: list[int] = []
    else:
        history_id = np.unique(
            np.linspace(0, episode_idx - 1, NUM_HISTORY, dtype=np.int32)
        ).tolist()
        placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_id)
        sources_value += f" These are your historical observations: {placeholder}."

    # The current frame is always appended last.
    sources_value += f" {CONJUNCTION}{DEFAULT_IMAGE_TOKEN}."

    n_images = len(history_id) + 1

    content: list[dict] = []
    for part in split_and_clean(sources_value):
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": None})  # placeholder
        else:
            content.append({"type": "text", "text": part})

    return [{"role": "user", "content": content}], n_images


def build_conversation_lookdown(
    episode_idx: int, instruction: str, assistant_reply: str
) -> list[dict]:
    """Build the turn-2 (look-down) conversation, matching the agent.

    The real flow is two turns: turn 1 returns '↓' (action 5 = look down); the robot tilts its
    camera; turn 2 sends the extra downward view **without resetting history**, and the coordinate
    is produced here. The turn-2 user message repeats no instruction/history — only
    " you can see <image>." — and only the newly appended look-down image binds to it.
    """
    turn1, _ = build_conversation(episode_idx, instruction)
    value = f" {CONJUNCTION}{DEFAULT_IMAGE_TOKEN}."

    content: list[dict] = []
    for part in split_and_clean(value):
        if part == DEFAULT_IMAGE_TOKEN:
            content.append({"type": "image", "image": None})
        else:
            content.append({"type": "text", "text": part})

    return [
        turn1[0],
        {"role": "assistant", "content": [{"type": "text", "text": assistant_reply}]},
        {"role": "user", "content": content},
    ]


def build_sample_inputs(sample: dict, processor):
    """Build processor ``inputs`` for one golden sample, handling both turn 1 and turn 2.

    Single source of truth for gate / calibration / profiling: every hand-rebuilt prompt is a
    place that can drift from real behavior. ``turn`` defaults to 1.
    """
    from PIL import Image

    turn = sample.get("turn", 1)
    images = [Image.open(p).convert("RGB") for p in sample["images"]]

    if turn == 2:
        reply = sample.get("assistant_turn1")
        if not reply:
            raise ValueError(
                f"Turn-2 sample missing 'assistant_turn1' "
                f"({sample['episode']} ep{sample['episode_idx']})."
            )
        conv = build_conversation_lookdown(
            sample["episode_idx"], sample["instruction"], reply)
    else:
        conv, _ = build_conversation(sample["episode_idx"], sample["instruction"])

    j = 0
    for t in conv:
        for it in t["content"]:
            if it["type"] == "image" and it.get("image") is None:
                it["image"] = images[j]
                j += 1
    if j != len(images):
        raise ValueError(
            f"<image> token count ({j}) != number of images ({len(images)}) — "
            f"{sample['episode']} ep{sample['episode_idx']} turn {turn}"
        )

    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, return_tensors="pt")


def census_one(processor, episode_idx: int, instruction: str) -> dict[str, Any]:
    """Count visual vs text tokens for one episode step (dummy images sized to the resize)."""
    conversation, n_images = build_conversation(episode_idx, instruction)

    images = [
        Image.fromarray(np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8)).convert("RGB")
        for _ in range(n_images)
    ]
    img_i = 0
    for item in conversation[0]["content"]:
        if item["type"] == "image":
            item["image"] = images[img_i]
            img_i += 1

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")

    ids = inputs["input_ids"][0]
    total = int(ids.numel())
    n_visual = int((ids == IMAGE_TOKEN_INDEX).sum())
    n_traj = int((ids == TRAJ_TOKEN_INDEX).sum())
    grid = inputs.get("image_grid_thw")
    return {
        "episode_idx": episode_idx,
        "n_images": n_images,
        "total_tokens": total,
        "visual_tokens": n_visual,
        "text_tokens": total - n_visual - n_traj,
        "traj_tokens": n_traj,
        "visual_pct": round(100.0 * n_visual / total, 2),
        "tokens_per_image": (n_visual // n_images) if n_images else 0,
        "image_grid_thw": grid.tolist() if grid is not None else None,
    }


def main() -> int:
    """Token census: report the visual/text token split the LLM actually sees at runtime."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model-path", default=os.environ.get(
        "INTERNVLA_CKPT", os.path.expanduser("~/InternNav/checkpoints/InternVLA-N1-DualVLN")))
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--out", default="token_census.json")
    args = ap.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"ERROR: model-path not found: {args.model_path}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {"model_path": args.model_path, "variants": {}}
    for variant, kwargs in PROCESSOR_VARIANTS.items():
        print(f"\n### processor variant: {variant}  {kwargs or '(default)'}")
        try:
            processor = AutoProcessor.from_pretrained(
                args.model_path, trust_remote_code=True, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not load processor: {exc}")
            report["variants"][variant] = {"error": str(exc)}
            continue
        rows = [census_one(processor, ep, SAMPLE_INSTRUCTION) for ep in args.episodes]
        for r in rows:
            print(f"  ep={r['episode_idx']:>3} imgs={r['n_images']:>2} "
                  f"total={r['total_tokens']:>6} visual={r['visual_tokens']:>6} "
                  f"visual%={r['visual_pct']:>6.2f}")
        report["variants"][variant] = {"processor_kwargs": kwargs, "rows": rows}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
