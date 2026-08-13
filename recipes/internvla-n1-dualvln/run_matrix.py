#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Collect every measurement for every built variant into one comparison table.

Reads what is already on disk rather than recomputing: checkpoint sizes, engine sizes,
the accuracy JSON from benchmark_accuracy.py, the NVFP4 investigation JSON, and the
z_latents numbers passed in. Anything missing prints as "-" instead of failing, so the
table can be produced at any point in the pipeline.

Emits Markdown, ready to paste into the recipe README.
"""
import argparse
import json
import os


def dir_size_gb(path: str) -> float | None:
    if not path or not os.path.isdir(path):
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


def file_size_gb(path: str) -> float | None:
    return os.path.getsize(path) / 1e9 if path and os.path.isfile(path) else None


def fmt(value, spec: str = ".2f", suffix: str = "") -> str:
    return "-" if value is None else f"{value:{spec}}{suffix}"


def load_json(path: str) -> dict | list | None:
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work_dir", default=os.path.expanduser("~/vln-opt-work"))
    p.add_argument("--output_path", default=None, help="Write the Markdown here")
    args = p.parse_args()
    w = args.work_dir

    # variant -> (checkpoint dir, engine dir, label)
    variants = [
        ("BF16 (unquantized)", "qwen25vl_system2", "engines/base_fp16"),
        ("FP8 s1", "qwen25vl_s1_fp8", "engines/s1_fp8"),
        ("NVFP4 s1 (experimental)", "qwen25vl_s1_nvfp4", "engines/s1_nvfp4"),
    ]

    acc = load_json(os.path.join(w, "out/accuracy_bf16_vs_fp8.json")) or []
    acc_by_path = {os.path.basename(r["model_path"]): r for r in acc}
    inv = load_json(os.path.join(w, "nvfp4_investigation.json")) or {}
    lat = load_json(os.path.join(w, "out/latency.json")) or {}
    zlat = load_json(os.path.join(w, "out/z_latents.json")) or {}

    rows = []
    for label, ckpt_name, eng_name in variants:
        ckpt = os.path.join(w, ckpt_name)
        eng_llm = os.path.join(w, eng_name, "llm/llm.engine")
        eng_vis = os.path.join(w, eng_name, "visual/visual.engine")
        a = acc_by_path.get(ckpt_name)
        rows.append({
            "label": label,
            "ckpt_gb": dir_size_gb(ckpt),
            "llm_gb": file_size_gb(eng_llm),
            "vis_gb": file_size_gb(eng_vis),
            "prefill_ms": lat.get(ckpt_name, {}).get("prefill_ms"),
            "decode_ms": lat.get(ckpt_name, {}).get("decode_ms"),
            "z_engine": zlat.get(ckpt_name),
            "z_weights": inv.get("bridge", {}).get(
                {"qwen25vl_s1_fp8": "fp8", "qwen25vl_s1_nvfp4": "nvfp4"}.get(ckpt_name, "")),
            "l2_mean": a and a.get("pixel_goal_l2_mean"),
            "l2_median": a and a.get("pixel_goal_l2_median"),
            "w_err": inv.get("weights", {}).get(
                {"qwen25vl_s1_fp8": "fp8", "qwen25vl_s1_nvfp4": "nvfp4"}.get(ckpt_name, ""),
                {}).get("rel_err_mean"),
        })

    out = []
    out.append("### Benchmark matrix\n")
    out.append("Every column is a real measurement on this machine; `-` means not measured.\n")
    out.append("| Variant | checkpoint | LLM engine | visual | prefill | decode "
               "| z_latents (engine) | z_latents (weights) | pixel L2 mean / median |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        l2 = ("-" if r["l2_mean"] is None
              else f"{r['l2_mean']:.2f} / {r['l2_median']:.2f} px")
        out.append(
            f"| {r['label']} | {fmt(r['ckpt_gb'], '.1f', ' GB')} | "
            f"{fmt(r['llm_gb'], '.2f', ' GB')} | {fmt(r['vis_gb'], '.2f', ' GB')} | "
            f"{fmt(r['prefill_ms'], '.1f', ' ms')} | {fmt(r['decode_ms'], '.1f', ' ms')} | "
            f"{fmt(r['z_engine'], '.6f')} | {fmt(r['z_weights'], '.6f')} | {l2} |")

    out.append("\n**Weight quantization error** (mean relative, 21 projections across "
               "layers 0/13/27):\n")
    out.append("| Variant | rel-err |")
    out.append("|---|---|")
    for r in rows:
        if r["w_err"] is not None:
            out.append(f"| {r['label']} | {100 * r['w_err']:.2f} % |")

    text = "\n".join(out)
    print(text)
    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
