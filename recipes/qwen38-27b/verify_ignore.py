#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
verify_ignore.py — audit our ignore list against a reference quantization config.

A reference config lists `modules_to_not_convert` by exact name, mixing modules
of every type. Most of those entries are norms, embeddings, Conv1d/Conv3d or
nn.Parameter, which `targets="Linear"` never touches anyway — so a raw set diff
is misleading. This walks the real module tree and reports only the entries that
matter: reference exclusions that ARE an nn.Linear present in this checkpoint but
absent from our list. Those are the genuine gaps.

    python verify_ignore.py --model_path <base> --reference ref_config.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_model import build_ignore_list, classify, load_model  # noqa: E402


def collapse(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", re.sub(r"\.\d+$", ".N", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--reference", required=True,
                    help="JSON holding quantization_config.modules_to_not_convert "
                         "(a full config.json works)")
    args = ap.parse_args()

    doc = json.load(open(args.reference))
    ref = doc.get("quantization_config", doc).get("modules_to_not_convert")
    if ref is None:
        raise SystemExit("No quantization_config.modules_to_not_convert in the reference")

    model, _ = load_model(args.model_path, real_weights=False)

    modules = dict(model.named_modules())
    linears = {n for n, m in modules.items() if isinstance(m, torch.nn.Linear)}
    ours = set(build_ignore_list(model))

    # Bucket every reference entry by why it does or does not matter here.
    genuine_gap, not_linear, absent, covered = [], [], [], []
    for name in ref:
        if name in ours:
            covered.append(name)
        elif name in linears:
            genuine_gap.append(name)
        elif name in modules:
            not_linear.append((name, type(modules[name]).__name__))
        else:
            absent.append(name)

    print("=" * 96)
    print("IGNORE-LIST AUDIT vs reference config")
    print("=" * 96)
    print(f"  reference entries          : {len(ref)}")
    print(f"  our ignore list            : {len(ours)} nn.Linear modules")
    print(f"  nn.Linear in this model    : {len(linears)}")
    print()
    print(f"  [OK]  already covered by us: {len(covered)}")
    print(f"  [OK]  not an nn.Linear     : {len(not_linear)}  (targets=\"Linear\" never reaches these)")
    print(f"  [OK]  absent from this ckpt: {len(absent)}")
    print(f"  [GAP] Linear, unprotected  : {len(genuine_gap)}")

    if not_linear:
        print("\n--- not an nn.Linear (by module class) ---")
        for cls, n in Counter(c for _, c in not_linear).most_common():
            example = next(nm for nm, c in not_linear if c == cls)
            print(f"  {cls:28} x{n:<4} e.g. {example}")

    if absent:
        print("\n--- named in the reference but absent from this checkpoint ---")
        for pattern, n in Counter(collapse(a) for a in absent).most_common():
            print(f"  {pattern:60} x{n}")

    print("\n--- what we quantize that the reference also quantizes ---")
    ref_set = set(ref)
    ours_quantized = sorted(linears - ours)
    also = [n for n in ours_quantized if n not in ref_set]
    for pattern, n in Counter(collapse(a) for a in also).most_common():
        print(f"  {pattern:60} x{n}")

    if genuine_gap:
        print("\n--- GAPS: reference protects these, we quantize them ---")
        for pattern, n in Counter(collapse(g) for g in genuine_gap).most_common():
            print(f"  {pattern:60} x{n}   role={classify(genuine_gap[0])}")
        print("\nRESULT: MISMATCH — review the policy table in inspect_model.py")
        sys.exit(1)

    print("\nRESULT: no gaps. Every reference exclusion that is an nn.Linear present "
          "in this checkpoint is already held at bf16.")


if __name__ == "__main__":
    main()
