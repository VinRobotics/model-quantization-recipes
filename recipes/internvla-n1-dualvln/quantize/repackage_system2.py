#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Repackage the InternVLA-N1 System 2 into a standalone Qwen2.5-VL checkpoint.

InternVLA-N1-DualVLN declares ``model_type: internvla_n1`` and ships no modeling code, so
it cannot be loaded with ``trust_remote_code`` -- the class has to come from the InternNav
repository. That dependency is avoidable for everything except System 1 itself, because the
System 2 backbone (vision tower + Qwen2.5 LLM) inside the checkpoint already uses standard
Qwen2.5-VL key names.

This is a lossless subset copy, streamed through ``safe_open`` so peak memory is one shard:

  keep    ``visual.*``, ``model.embed_tokens``, ``model.norm``, ``model.layers.*``, ``lm_head``
  drop    the System 1 modules (traj_dit, rgb_model, rgb_resampler, memory_encoder,
          cond_projector, action_encoder, action_decoder, pos_encoding)
  rewrite ``config.json`` to ``model_type=qwen2_5_vl`` /
          ``architectures=[Qwen2_5_VLForConditionalGeneration]``

Weights are copied bit-for-bit; the source is opened read-only.

Two System 1 tensors are *not* simply dropped. ``latent_queries`` and ``cond_projector`` form
the System 2 -> System 1 bridge, and the fidelity checks need them to compute z_latents. They
are written to a separate ``bridge.safetensors`` (~25 MB) so that quantization, export, engine
build and verification can all run from this directory alone, without reopening the 16 GB
source checkpoint and without importing InternNav.
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

SYSTEM1_PREFIXES = (
    "model.traj_dit",
    "model.rgb_model",
    "model.rgb_resampler",
    "model.memory_encoder",
    "model.cond_projector",
    "model.action_encoder",
    "model.action_decoder",
    "model.pos_encoding",
)

# Tensors that belong to System 1 but are needed to evaluate the System 2 -> System 1
# bridge. Kept aside rather than dropped; see the module docstring.
BRIDGE_KEYS = ("model.latent_queries", "model.cond_projector")

# Non-weight files carried over verbatim. Matching by prefix rather than an exact list
# keeps this working when a checkpoint ships an extra tokenizer artifact.
COPY_PREFIXES = (
    "tokenizer",
    "vocab",
    "merges",
    "preprocessor",
    "chat_template",
    "generation_config",
    "added_tokens",
    "special_tokens",
)

# A repackaged 7B System 2 is ~15.5 GB. Refuse to start rather than die at 90 %.
MIN_FREE_GB = 18


def is_system2(key: str) -> bool:
    """Keep the standard vision + LLM keys; drop every System 1 module."""
    if key.startswith(SYSTEM1_PREFIXES):
        return False
    return (
        key.startswith("visual.")
        or key == "lm_head.weight"
        or key.startswith("model.embed_tokens")
        or key.startswith("model.norm")
        or key.startswith("model.layers.")
    )


def is_bridge(key: str) -> bool:
    """Tensors kept aside for z_latents evaluation."""
    return key.startswith(BRIDGE_KEYS)


def build_config(src: str) -> dict:
    """Rewrite the InternVLA config into a plain Qwen2.5-VL one."""
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    for key in ("system1", "n_query", "model_cfg", "model_type", "architectures", "auto_map"):
        cfg.pop(key, None)
    cfg["model_type"] = "qwen2_5_vl"
    cfg["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    return cfg


def free_gb(path: str) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True,
                        help="Source InternVLA-N1-DualVLN checkpoint directory")
    parser.add_argument("--output_path", required=True,
                        help="Destination directory for the stock Qwen2.5-VL System 2 checkpoint")
    parser.add_argument("--free_source", action="store_true",
                        help="Delete each source shard once its converted copy is written. "
                             "Peak disk becomes one shard rather than two checkpoints. "
                             "Destructive -- only use this if the source can be re-downloaded.")
    parser.add_argument("--skip_disk_check", action="store_true",
                        help="Skip the free-space preflight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src, dst = args.model_path, args.output_path

    if not os.path.isdir(src):
        print(f"[ERROR] --model_path does not exist: {src}")
        return 1
    index_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(f"[ERROR] no model.safetensors.index.json under {src}; "
              f"a sharded checkpoint is expected")
        return 1

    os.makedirs(dst, exist_ok=True)

    if not args.skip_disk_check and not args.free_source:
        avail = free_gb(dst)
        if avail < MIN_FREE_GB:
            print(f"[ERROR] only {avail:.1f} GB free under {dst}; need >= {MIN_FREE_GB} GB. "
                  f"Free space, point --output_path elsewhere, or pass --free_source to "
                  f"delete each source shard as it is converted.")
            return 1

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    kept = [k for k in weight_map if is_system2(k)]
    bridge = [k for k in weight_map if is_bridge(k)]
    dropped = [k for k in weight_map if not is_system2(k)]
    print(f"keep {len(kept)} System2 keys, drop {len(dropped)} System1 keys "
          f"({len(bridge)} of them kept aside as bridge tensors)")

    # Group by source shard so each is opened once and read sequentially.
    by_shard: dict[str, list[str]] = {}
    for key in kept:
        by_shard.setdefault(weight_map[key], []).append(key)
    bridge_by_shard: dict[str, list[str]] = {}
    for key in bridge:
        bridge_by_shard.setdefault(weight_map[key], []).append(key)

    new_weight_map: dict[str, str] = {}
    bridge_tensors: dict[str, "object"] = {}
    total_bytes = 0
    out_shards = sorted(by_shard)
    n_shards = len(out_shards)

    for i, shard in enumerate(out_shards, 1):
        out_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for key in by_shard[shard]:
                tensor = f.get_tensor(key)
                tensors[key] = tensor
                total_bytes += tensor.numel() * tensor.element_size()
                new_weight_map[key] = out_name
            for key in bridge_by_shard.get(shard, []):
                bridge_tensors[key] = f.get_tensor(key)
        save_file(tensors, os.path.join(dst, out_name), metadata={"format": "pt"})
        print(f"  [{i}/{n_shards}] {out_name}: {len(tensors)} tensors")
        del tensors
        if args.free_source:
            os.remove(os.path.join(src, shard))
            print(f"        removed source shard {shard}")

    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
                  f, indent=2)

    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(build_config(src), f, indent=2)

    if bridge_tensors:
        save_file(bridge_tensors, os.path.join(dst, "bridge.safetensors"),
                  metadata={"format": "pt"})
        print(f"  bridge.safetensors: {len(bridge_tensors)} tensors "
              f"({', '.join(sorted(bridge_tensors))})")
    else:
        print("[WARN] no bridge tensors found; z_latents verification will need the "
              "original checkpoint")

    for name in os.listdir(src):
        if name.endswith(".safetensors"):
            continue
        if any(name.startswith(p) for p in COPY_PREFIXES):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    dropped_modules = sorted({k.split(".")[1] for k in dropped if "." in k})
    print(f"\nDone -> {dst}")
    print(f"  {n_shards} shards, {total_bytes / 1e9:.1f} GB System 2")
    print("  config model_type=qwen2_5_vl")
    print(f"  dropped System 1 modules: {dropped_modules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
