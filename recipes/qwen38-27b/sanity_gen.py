#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
sanity_gen.py — does this checkpoint still produce coherent text?

A broken quantization usually shows up as repetition or token soup long before
a benchmark score moves, so run this before spending GPU-hours on lm-eval.

    python sanity_gen.py --ckpt outputs/qwen38-27b-fp8-dynamic
"""

import argparse

from vllm import LLM, SamplingParams

_PROMPTS = [
    "The capital of France is",
    "Q: If a train travels 60 km in 45 minutes, what is its average speed in km/h?\nA:",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Summarize in one sentence: The mitochondrion is an organelle found in most "
    "eukaryotic cells that generates most of the cell's supply of ATP.\nSummary:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=96)
    # Every decode sequence in a Gated DeltaNet layer holds one Mamba-style state
    # block, so concurrency is capped by state memory, not KV cache. vLLM's
    # default max_num_seqs=1024 exceeds what fits and aborts CUDA graph capture
    # with "max_num_seqs exceeds available Mamba cache blocks".
    ap.add_argument("--max_num_seqs", type=int, default=64)
    args = ap.parse_args()

    llm = LLM(
        model=args.ckpt,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    # Greedy: any nondeterminism here would just muddy the comparison.
    outs = llm.generate(_PROMPTS, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    print("\n" + "=" * 90)
    print(f"SANITY GENERATION — {args.ckpt}")
    print("=" * 90)
    for out in outs:
        print(f"\n>>> {out.prompt!r}")
        print(f"    {out.outputs[0].text.strip()!r}")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
