#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Isolate the loss that appears only inside the engine.

NVFP4 measures 0.9786 as fake quant in PyTorch and 0.9310 through the engine. Activation
quantization accounts for 0.009 of the total; the other 0.048 shows up only once the engine
runs, and FP8 has nothing comparable (0.006 in total). This measures that engine-specific
error rather than inferring it from two numbers taken in different environments.

The reference is what makes it work. Every other check here compares against the
*unquantized* model, which folds quantization error and engine error into one figure. This
one runs the fake-quant model and the engine **in the same process against the same
inputs**, so whatever separates them is the engine alone: kernel selection, block-scale
arithmetic, accumulation order, or a miscompile.

Read the output as:

  cosine ~1.0     the engine reproduces correct W4A4 and the loss is quantization after all
  cosine ~0.95    the engine diverges materially -- the 0.048, localised
  cosine << 0.9   the NVFP4 kernel path is badly wrong

Pass an FP8 engine as a control. FP8 should sit near 1.0; if it does not, suspect the
harness before the NVFP4 path.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "quantize"))  # quant_schemes

import engine_runner as er  # noqa: E402

PROMPT = ("You are an autonomous navigation assistant. Your task is to go to the kitchen. "
          "Where should you go next to stay on track?")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def build_fake_quant(base_ckpt: str, scheme: str, strategy: str, n_calib: int, device: str):
    import modelopt.torch.quantization as mtq
    from quant_schemes import build_quant_config
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    processor = AutoProcessor.from_pretrained(base_ckpt, min_pixels=128 * 28 * 28,
                                              max_pixels=2048 * 32 * 32)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()

    texts = [PROMPT] * n_calib

    def forward_loop(m):
        for t in texts:
            with torch.no_grad():
                m(tokenizer(t, return_tensors="pt")["input_ids"].to(device))

    model = mtq.quantize(model, build_quant_config(scheme, strategy),
                         forward_loop=forward_loop)
    return model.eval(), tokenizer, processor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repkg_ckpt", required=True)
    p.add_argument("--engine_path", required=True)
    p.add_argument("--scheme", default="nvfp4_default")
    p.add_argument("--strategy", default="s1")
    p.add_argument("--num_calib_samples", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[1/3] fake quant in PyTorch ({args.scheme})")
    model, tokenizer, _ = build_fake_quant(args.repkg_ckpt, args.scheme, args.strategy,
                                           args.num_calib_samples, args.device)

    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(args.device)
    inner = model.model
    embed = inner.get_input_embeddings() if hasattr(inner, "get_input_embeddings") \
        else model.get_input_embeddings()
    embeds = embed(ids)

    # The engine emits hidden states *before* the final norm, while
    # output_hidden_states[-1] is the post-norm tensor. Comparing the two reads ~0.49 for
    # a known-good FP8 engine, so hook the norm and take its input instead. (Found by
    # running the FP8 control, which is why the control is not optional.)
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    captured = {}
    handle = lm.norm.register_forward_hook(
        lambda mod, inp, out: captured.update(pre=inp[0].detach()))
    with torch.inference_mode():
        inner(inputs_embeds=embeds, use_cache=False)
    handle.remove()
    h_pt = captured["pre"][0].float().cpu()
    print(f"      hidden states (pre-norm) {tuple(h_pt.shape)}")

    del model
    torch.cuda.empty_cache()

    print(f"[2/3] engine {args.engine_path}")
    os.environ["ENGINE_PATH"] = args.engine_path
    seq = embeds.shape[1]
    pos = torch.arange(seq, device=args.device).view(1, 1, -1).expand(3, 1, -1)
    rope = er.build_mrope_table(pos, args.device)
    _, hidden = er.run_engine(embeds.half(), rope)
    h_eng = hidden[0].float().cpu()

    print("[3/3] compare")
    n = min(h_pt.shape[0], h_eng.shape[0])
    per_tok = [cos(h_pt[i], h_eng[i]) for i in range(n)]
    print("\n=== engine vs fake quant, same weights and same activation quantization ===")
    print(f"  full-sequence cosine : {cos(h_pt[:n], h_eng[:n]):.6f}")
    print(f"  last token           : {per_tok[-1]:.6f}")
    print(f"  worst token          : {min(per_tok):.6f} (position {per_tok.index(min(per_tok))})")
    print("\n  Near 1.0 means the engine reproduces correct W4A4 and the loss lies in")
    print("  quantization. Materially below means the divergence is the engine's own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
