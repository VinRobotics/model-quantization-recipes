#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Build a navigation-prompt calibration JSONL for `tensorrt-edgellm-quantize`.

The native `tensorrt-edgellm-quantize llm --text_dataset NAME` flag only accepts a
*registered* dataset name (the default is `cnn_dailymail`), but any built-in can be pointed
at a local file with `EDGELLM_QUANT_DATASET_<NAME>=/path/to/file.jsonl` -- see
`tensorrt_edgellm/quantization/datasets/__init__.py::local_override_path`. This script writes
that file in the schema the override loader expects: one `{"article": "..."}` object per line
(the field name `cnn_dailymail()` reads, since that is the dataset being overridden).

Why bother, instead of the CLI's cnn_dailymail default: calibrating on the deployment prompt's
own domain measurably changes activation scales. The same FP8 recipe went from trajectory
cosine 0.909 (cnn_dailymail) to 0.978 (this) in earlier testing -- calibration text that never
resembles a navigation instruction leaves the backbone's activation ranges fitted to news
articles instead of the prompts it actually deploys on.

One thing this does *not* need, unlike an earlier version of this recipe: the four trailing
trajectory-query tokens (`<|latent_q0..3|>`). Those only matter for calibrating the
System-2 -> System-1 bridge forward pass, and `tensorrt-edgellm-quantize` never runs that --
it loads System 2 as a stock Qwen2.5-VL (see `internvla_n1_loader.py` in TensorRT-Edge-LLM)
and calibrates with ordinary text forward passes. Appending tokens the tokenizer does not yet
know about (they are registered later, at export time) would only add noise.

Usage:
    huggingface-cli download InternRobotics/InternData-N1 \\
        vln_ce/raw_data/r2r/train/train.json.gz --repo-type dataset \\
        --local-dir $CALIB_DATA_ROOT
    python build_calib_jsonl.py \\
        --train_json $CALIB_DATA_ROOT/vln_ce/raw_data/r2r/train/train.json.gz \\
        --output $CALIB_DATA_ROOT/nav_calib.jsonl
"""
import argparse
import gzip
import json
import random

PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to {instruction} "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed "
    "the task.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_json", required=True,
                    help="InternData-N1 vln_ce/raw_data/r2r/train/train.json.gz")
    ap.add_argument("--output", required=True, help="Destination JSONL")
    ap.add_argument("--num_samples", type=int, default=512,
                    help="Matches the native CLI's calibration sample count (default: 512)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with gzip.open(args.train_json) as f:
        episodes = json.load(f)["episodes"]

    random.Random(args.seed).shuffle(episodes)

    written = 0
    with open(args.output, "w") as out:
        for ep in episodes:
            if written >= args.num_samples:
                break
            text = ep["instruction"]["instruction_text"].strip()
            if not text:
                continue
            instruction = text.rstrip(". ")
            prompt = PROMPT_TEMPLATE.format(instruction=instruction + ".")
            out.write(json.dumps({"article": prompt}) + "\n")
            written += 1

    print(f"wrote {written} navigation calibration prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
