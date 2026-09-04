#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
inspect_model.py — architecture census for Qwen3.8-27B before quantization.

Instantiates the model on the *meta* device from config alone, so the full
module tree is available without reading 54 GiB of bf16 weights. Prints:

  1. print(model)                — the raw module tree
  2. a per-Linear census         — every nn.Linear, grouped by role
  3. a recommended ignore list   — which modules to keep in bf16, and why

The ignore list this emits is the same one quantize.py builds at runtime, so
this script is the audit trail for the quantization decisions.

Usage:
    python inspect_model.py --model_path /path/to/Qwen3.8-27B
    python inspect_model.py --model_path ... --real-weights   # load for real
"""

import argparse
import json
import re
from collections import OrderedDict, defaultdict

import torch
import transformers
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Role classification
#
# Qwen3.8-27B (model_type "qwen3_5") is a dense VLM with a hybrid text stack:
#   64 layers laid out as 16 x (3 x GatedDeltaNet -> FFN, 1 x GatedAttention -> FFN)
# The `linear_attention` layers are stateful SSM blocks (Gated DeltaNet); the
# `full_attention` layers are ordinary gated attention. They need different
# treatment, so the census separates them.
# ---------------------------------------------------------------------------

_ROLE_RULES = [
    # (role, regex over the module's dotted name) — first match wins, so the
    # narrow SSM rules must precede the catch-all ones.
    # `deepstack_merger_list` is empty on this checkpoint (deepstack_visual_indexes
    # is []), but it is part of the vision tower wherever it is populated.
    ("vision",        r"(^|\.)(visual|vision_tower|vision_model)\."),
    ("lm_head",       r"(^|\.)lm_head$"),
    # MTP is not instantiated by Qwen3_5ForConditionalGeneration in transformers
    # 5.14.1, but the official config excludes mtp.fc, so cover it if it appears.
    ("mtp",           r"(^|\.)mtp\b"),
    # MoE router / shared-expert gates. Absent from this dense checkpoint; kept
    # so the policy transfers to the MoE sibling (Qwen3_5MoeForConditionalGeneration).
    ("moe_gate",      r"mlp\.(gate|shared_expert_gate)$"),
    # Gated DeltaNet: in_proj_a / in_proj_b emit the per-head decay and delta-rule
    # beta scalars (5120 -> 48). Separate role from the bulk projections.
    # in_proj_ba is the fused a+b variant some Qwen3.5-family checkpoints ship;
    # the official quantization config excludes it alongside a and b.
    ("ssm_decay",     r"linear_attn\..*in_proj_(a|b|ba)$"),
    ("ssm_in",        r"linear_attn\..*in_proj"),
    ("ssm_out",       r"linear_attn\..*out_proj"),
    ("ssm_other",     r"linear_attn\."),
    ("attn_qkv",      r"self_attn\..*(q_proj|k_proj|v_proj)$"),
    ("attn_out",      r"self_attn\..*o_proj$"),
    ("attn_other",    r"self_attn\."),
    ("mlp_up",        r"mlp\..*(gate_proj|up_proj)$"),
    ("mlp_down",      r"mlp\..*down_proj$"),
    ("mlp_other",     r"mlp\."),
]

_ROLE_ORDER = [
    "attn_qkv", "attn_out", "attn_other",
    "ssm_in", "ssm_out", "ssm_decay", "ssm_other",
    "mlp_up", "mlp_down", "moe_gate", "mlp_other",
    "vision", "mtp", "lm_head", "other",
]

# How each role should be treated, and the reason. Consumed by quantize.py.
_POLICY = OrderedDict([
    ("attn_qkv",   ("quantize", "dense GEMM, well-conditioned per-channel — the bulk of attention FLOPs")),
    ("attn_out",   ("quantize", "dense GEMM, post-softmax activations are bounded")),
    ("mlp_up",     ("quantize", "largest GEMMs in the model (5120 -> 17408), dominates weight footprint")),
    ("mlp_down",   ("quantize", "large GEMM (17408 -> 5120); SwiGLU output outliers are handled by "
                                "per-channel scales")),
    ("ssm_out",    ("quantize", "plain projection out of the SSM, no recurrent state on this path")),
    ("ssm_in",     ("quantize", "in_proj_qkv / in_proj_z are ordinary dense GEMMs (4.0B params, 16% of "
                                "all Linear weight); their output is consumed elementwise per position, "
                                "so error stays local rather than entering the recurrence")),
    ("ssm_decay",  ("keep_bf16", "in_proj_a / in_proj_b set the per-head decay and delta-rule beta that "
                                 "drive the recurrence. Error here compounds multiplicatively over the "
                                 "whole sequence, and at 5120x48 they are 0.09% of Linear weight — the "
                                 "cheapest possible place to spend precision")),
    ("ssm_other",  ("keep_bf16", "conservative default for unclassified Gated DeltaNet internals")),
    ("attn_other", ("keep_bf16", "q_norm/k_norm and gates — tiny, and normalization is precision-critical")),
    ("moe_gate",   ("keep_bf16", "MoE router / shared-expert gate — a wrong routing decision costs a whole "
                                 "expert, and the module is tiny. Absent from this dense checkpoint")),
    ("mlp_other",  ("keep_bf16", "unclassified MLP submodules — tiny parameter count, high sensitivity")),
    ("vision",     ("keep_bf16", "ViT is ~0.5B of 27B; quantizing it buys almost nothing and vision "
                                 "tokens drive every downstream answer")),
    ("mtp",        ("keep_bf16", "multi-token-prediction head is a speculative-decoding side path")),
    ("lm_head",    ("keep_bf16", "248k-row output projection straight into the softmax — the single "
                                 "most accuracy-sensitive layer in the model")),
    ("other",      ("keep_bf16", "unclassified — default to safe")),
])


def classify(name: str) -> str:
    for role, pattern in _ROLE_RULES:
        if re.search(pattern, name):
            return role
    return "other"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_model_class(config):
    """Pick the concrete class named in config.architectures.

    Qwen3.8-27B is a `Qwen3_5ForConditionalGeneration`. Going through
    AutoModelForCausalLM silently yields `Qwen3_5ForCausalLM` — the text stack
    only — which hides the vision tower from the census. Resolve by name.
    """
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            print(f"Model class: {arch}")
            return cls
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, auto_name, None)
        if cls is not None:
            print(f"[WARN] config.architectures unusable; falling back to {auto_name}")
            return cls
    raise RuntimeError("Could not resolve a model class for this checkpoint")


def load_model(model_path: str, real_weights: bool):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls = resolve_model_class(config)
    if real_weights:
        print("Loading real weights (this reads ~54 GiB) ...")
        return cls.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        ), config

    print("Instantiating on meta device from config (no weight I/O) ...")
    with torch.device("meta"):
        model = cls._from_config(config) if hasattr(cls, "_from_config") \
            else cls.from_config(config)
    return model, config


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _numel(module) -> int:
    # Meta tensors have shapes but no storage; numel() still works.
    return sum(p.numel() for p in module.parameters(recurse=False))


def census(model):
    """Return {role: {"count": n, "params": n, "examples": [...], "shapes": {...}}}."""
    stats = defaultdict(lambda: {"count": 0, "params": 0, "examples": [], "shapes": defaultdict(int)})
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        role = classify(name)
        s = stats[role]
        s["count"] += 1
        s["params"] += _numel(module)
        s["shapes"][f"{module.in_features}x{module.out_features}"] += 1
        if len(s["examples"]) < 2:
            s["examples"].append(name)
    return stats


def print_census(stats):
    total_lin = sum(s["params"] for s in stats.values())
    print("\n" + "=" * 100)
    print("LINEAR-LAYER CENSUS  (nn.Linear only; norms/embeddings excluded)")
    print("=" * 100)
    print(f"{'role':<12}{'#mods':>7}{'params':>16}{'% of Linear':>13}  {'action':<11} example")
    print("-" * 100)
    q_params = 0
    for role in _ROLE_ORDER:
        if role not in stats:
            continue
        s = stats[role]
        action = _POLICY[role][0]
        if action == "quantize":
            q_params += s["params"]
        pct = 100 * s["params"] / total_lin if total_lin else 0
        ex = s["examples"][0] if s["examples"] else ""
        print(f"{role:<12}{s['count']:>7}{s['params']:>16,}{pct:>12.2f}%  {action:<11} {ex}")
    print("-" * 100)
    print(f"{'TOTAL':<12}{sum(s['count'] for s in stats.values()):>7}{total_lin:>16,}")
    print(f"\nCovered by quantization: {q_params:,} params "
          f"({100 * q_params / total_lin:.2f}% of all Linear weight)")
    print(f"Held at bf16           : {total_lin - q_params:,} params "
          f"({100 * (total_lin - q_params) / total_lin:.2f}%)")


def print_policy():
    print("\n" + "=" * 100)
    print("POLICY — why each role is treated the way it is")
    print("=" * 100)
    for role, (action, reason) in _POLICY.items():
        print(f"  [{action:<10}] {role:<12} {reason}")


def print_shapes(stats):
    print("\n" + "=" * 100)
    print("SHAPE BREAKDOWN  (in_features x out_features -> count)")
    print("=" * 100)
    for role in _ROLE_ORDER:
        if role not in stats:
            continue
        shapes = ", ".join(f"{k}x{v}" for k, v in sorted(
            stats[role]["shapes"].items(), key=lambda kv: -kv[1]))
        print(f"  {role:<12} {shapes}")


def build_ignore_list(model) -> list:
    """The exact `ignore=` list quantize.py will hand to llmcompressor.

    Derived from the live module tree rather than hardcoded indices, so it
    tracks the architecture instead of assuming it.
    """
    keep = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and _POLICY[classify(name)][0] == "keep_bf16":
            keep.add(name)
    return sorted(keep)


def summarize_ignore(names: list) -> list:
    """Collapse per-layer names into readable regexes for the recipe."""
    patterns = []
    seen = set()
    for n in names:
        # model.language_model.layers.37.mlp.gate -> re:.*\.mlp\.gate$
        generic = re.sub(r"\.\d+\.", ".*.", n)
        if generic not in seen:
            seen.add(generic)
            patterns.append(generic)
    return patterns


def main():
    ap = argparse.ArgumentParser(description="Architecture census for Qwen3.8-27B")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--real-weights", action="store_true",
                    help="Load actual weights instead of meta device (slow, needs GPU)")
    ap.add_argument("--dump-ignore", default="",
                    help="Write the resolved ignore list to this JSON path")
    ap.add_argument("--no-tree", action="store_true", help="Skip the raw print(model) dump")
    args = ap.parse_args()

    model, config = load_model(args.model_path, args.real_weights)

    print("\n" + "=" * 100)
    print("CONFIG SUMMARY")
    print("=" * 100)
    tc = getattr(config, "text_config", config)
    vc = getattr(config, "vision_config", None)
    print(f"  architectures        : {getattr(config, 'architectures', None)}")
    print(f"  model_type           : {getattr(config, 'model_type', None)}")
    print(f"  text hidden_size     : {getattr(tc, 'hidden_size', '?')}")
    print(f"  text intermediate    : {getattr(tc, 'intermediate_size', '?')}")
    print(f"  text num_layers      : {getattr(tc, 'num_hidden_layers', '?')}")
    print(f"  vocab_size           : {getattr(tc, 'vocab_size', '?')}")
    lt = getattr(tc, "layer_types", None)
    if lt:
        from collections import Counter
        print(f"  layer_types          : {dict(Counter(lt))}")
    if vc is not None:
        print(f"  vision depth/hidden  : {getattr(vc, 'depth', '?')} / {getattr(vc, 'hidden_size', '?')}")

    if not args.no_tree:
        print("\n" + "=" * 100)
        print("MODULE TREE — print(model)")
        print("=" * 100)
        print(model)

    stats = census(model)
    print_census(stats)
    print_shapes(stats)
    print_policy()

    ignore = build_ignore_list(model)
    print("\n" + "=" * 100)
    print(f"RESOLVED IGNORE LIST — {len(ignore)} modules held at bf16")
    print("=" * 100)
    for p in summarize_ignore(ignore):
        print(f"  {p}")

    if args.dump_ignore:
        with open(args.dump_ignore, "w") as f:
            json.dump({"ignore": ignore, "patterns": summarize_ignore(ignore)}, f, indent=2)
        print(f"\nWrote ignore list -> {args.dump_ignore}")


if __name__ == "__main__":
    main()
