#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Quantization-aware fine-tuning for the InternVLA-N1 System 2 planner.

Why this exists. Post-training NVFP4 leaves the System 2 -> System 1 bridge at z_latents
0.931 through the engine, under the 0.99 gate that FP8 clears. Weight quantization is not
what costs it -- NVFP4 weights alone measure 0.988 -- so the remaining loss sits in the
4-bit *activations*, which no amount of weight-side scaling can fix after the fact. QAT is
the standard answer: let the model see the quantization noise during training and adapt to
it.

What this does. ModelOpt has no separate QAT entry point; QAT is ``mtq.quantize`` followed
by ordinary fine-tuning, with the fake-quantizers left in place so gradients flow through
them (straight-through estimation). So this:

  1. loads the repackaged System 2,
  2. calibrates and inserts quantizers exactly as ``quantize.py`` would,
  3. fine-tunes on real VLN episodes with the deployed prompt,
  4. exports through the same path, producing a checkpoint the existing engine build and
     verification scripts accept unchanged.

Read the result honestly. **Success rate is not measurable here** -- SR, SPL and NE are all
closed-loop and need Habitat or InternUtopia, neither of which runs on a Jetson. What this
optimises and what you can check is the proxy: z_latents cosine and pixel-goal L2 on
held-out episodes. A proxy that improves is a necessary condition for SR to improve, not
evidence that it did.

Train/eval separation. The calibration and probe sets share one MP3D scene
(``YmJkqBEsHnH`` appears as ``calib_scenes/r2r/`` and ``probe_heldout/rxr/`` -- same
building, different split), and the source project has already been bitten by exactly this
overlap once, reporting a calibration gain that turned out to be leakage. That scene is
excluded from training by default; ``--allow_overlap`` turns the guard off if you want to
measure the effect of the leak deliberately.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_builder as pb  # noqa: E402
from model_loader import export_quantized_model, load_model  # noqa: E402
from quant_schemes import build_quant_config, calib_batch_size, validate  # noqa: E402

# Same building as probe_heldout/rxr/YmJkqBEsHnH. Training on it contaminates the
# held-out evaluation even though the episodes and instructions differ.
OVERLAPPING_SCENES = ("YmJkqBEsHnH",)


def discover_training_samples(data_root: str, max_samples: int, camera: str,
                              allow_overlap: bool, seed: int = 0) -> list[dict]:
    """Collect prompt/target pairs from LeRobot episodes, skipping leaked scenes."""
    import glob
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rgb_key = f"observation.images.rgb.{camera}"
    level_key = "observation.images.rgb.125cm_0deg"
    goal_col = f"goal.{camera}"
    samples: list[dict] = []
    skipped_scenes: set[str] = set()

    for meta in sorted(glob.glob(os.path.join(data_root, "**", "meta", "episodes.jsonl"),
                                 recursive=True)):
        scene_dir = os.path.dirname(os.path.dirname(meta))
        scene = os.path.basename(scene_dir)
        if not allow_overlap and scene in OVERLAPPING_SCENES:
            skipped_scenes.add(scene)
            continue

        for ep in (json.loads(line) for line in open(meta)):
            idx, length = ep["episode_index"], ep["length"]
            table = None
            for parquet in glob.glob(os.path.join(scene_dir, "data", "**",
                                                  f"episode_{idx:06d}.parquet"),
                                     recursive=True):
                table = pq.read_table(parquet)
                break
            if table is None or goal_col not in table.schema.names:
                continue

            goals = table.column(goal_col).to_pylist()
            usable = [i for i in range(1, min(length, len(goals)))
                      if goals[i] is not None and int(goals[i][0]) >= 0]
            if not usable:
                continue
            t = int(usable[rng.integers(0, len(usable))])

            history = np.unique(np.linspace(0, t - 1, pb.NUM_HISTORY, dtype=np.int32)).tolist()
            frames = [os.path.join(scene_dir, "videos", "chunk-000", level_key,
                                   f"episode_{idx:06d}_{i}.jpg") for i in history + [t]]
            lookdown = os.path.join(scene_dir, "videos", "chunk-000", rgb_key,
                                    f"episode_{idx:06d}_{t}.jpg")
            if not all(os.path.isfile(p) for p in frames + [lookdown]):
                continue

            gt = goals[t]
            samples.append({
                "episode": scene,
                "episode_idx": t,
                "instruction": (ep.get("tasks") or [""])[0],
                "images": frames + [lookdown],
                "turn": 2,
                "assistant_turn1": "↓",
                # The supervision target is the deployed answer format: "row col".
                "target": f"{int(gt[0])} {int(gt[1])}",
            })
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    if skipped_scenes:
        print(f"      [data] excluded {sorted(skipped_scenes)} -- also present in the "
              f"held-out probe set")
    return samples


def build_batch(sample: dict, processor, device: str):
    """Prompt plus target, with the prompt masked out of the loss."""
    inputs = pb.build_sample_inputs(sample, processor)
    prompt_len = inputs["input_ids"].shape[1]

    target_ids = processor.tokenizer(sample["target"], add_special_tokens=False,
                                     return_tensors="pt")["input_ids"]
    input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100          # supervise the answer only

    batch = {k: v for k, v in inputs.items() if k != "input_ids"}
    batch["input_ids"] = input_ids
    batch["labels"] = labels
    if "attention_mask" in batch:
        pad = torch.ones((1, target_ids.shape[1]), dtype=batch["attention_mask"].dtype)
        batch["attention_mask"] = torch.cat([batch["attention_mask"], pad], dim=1)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def freeze_all_but_last_layers(model, n_last: int) -> int:
    """Freeze everything except the final ``n_last`` decoder layers.

    Returns the number of frozen parameters. Full fine-tuning does not fit: at 8.29B
    parameters, weights plus gradients plus AdamW moments come to about 100 GB before any
    activation memory, against a 122 GB pool shared with the host. Restricting to the last
    layers is also where the quantity being repaired lives -- the System 1 bridge reads the
    last layer's hidden states.
    """
    layers = None
    for owner in (getattr(model, "model", None), model):
        inner = getattr(owner, "language_model", owner)
        if inner is not None and hasattr(inner, "layers"):
            layers = inner.layers
            break
    if layers is None:
        raise AttributeError("could not locate the decoder layer list on this model")

    keep = {id(p) for layer in layers[-n_last:] for p in layer.parameters()}
    frozen = 0
    for param in model.parameters():
        if id(param) not in keep:
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="Repackaged System 2 checkpoint")
    p.add_argument("--output_path", required=True)
    p.add_argument("--data_root", required=True, help="LeRobot episodes for training")
    p.add_argument("--scheme", default="nvfp4_default")
    p.add_argument("--strategy", default="s1")
    p.add_argument("--num_train_samples", type=int, default=64)
    p.add_argument("--num_calib_samples", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Small on purpose: QAT adapts to quantization noise, it does not "
                        "re-learn the task, and a large step undoes the pretrained planner")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--train_last_n_layers", type=int, default=4,
                   help="Train only the last N decoder layers; freeze the rest. Full "
                        "fine-tuning of the 8.29B planner needs ~100 GB for weights, "
                        "gradients and AdamW state alone, which the 122 GB unified pool "
                        "cannot hold alongside a 10-image prompt's activations. The bridge "
                        "reads the last layer's hidden states, so the last layers are also "
                        "where the signal it depends on is formed. 0 trains everything.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--camera", default="125cm_30deg")
    p.add_argument("--allow_overlap", action="store_true",
                   help="Permit training on scenes that also appear in the probe set")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    import modelopt.torch.quantization as mtq

    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        validate(args.scheme, args.strategy, model_path=args.model_path,
                 allow_experimental=True)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[1/5] Loading {args.model_path}")
    model, tokenizer, processor = load_model(args.model_path, dtype="bf16",
                                             device=args.device)
    if processor is None:
        print("[ERROR] no processor; VLN prompts cannot be built")
        return 1

    print(f"[2/5] Collecting training samples from {args.data_root}")
    train = discover_training_samples(args.data_root, args.num_train_samples,
                                      args.camera, args.allow_overlap, args.seed)
    if not train:
        print("[ERROR] no usable training samples")
        return 1
    print(f"      {len(train)} samples from "
          f"{len({s['episode'] for s in train})} scene(s)")

    print(f"[3/5] Inserting quantizers ({args.scheme} / {args.strategy})")
    quant_cfg = build_quant_config(args.scheme, args.strategy)
    calib = train[:min(args.num_calib_samples, len(train))]

    def forward_loop(m):
        for s in calib:
            with torch.no_grad():
                m(**build_batch(s, processor, args.device))

    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    _ = calib_batch_size(args.scheme, is_image_calib=True)

    print(f"[4/5] Fine-tuning: {args.epochs} epoch(s), lr={args.lr}, "
          f"grad_accum={args.grad_accum}")
    # Gradients flow through the fake-quantizers by straight-through estimation, which is
    # what lets the weights adapt to 4-bit activation noise rather than merely to 4-bit
    # weights.
    model.train()
    model.config.use_cache = False

    if args.train_last_n_layers > 0:
        n_frozen = freeze_all_but_last_layers(model, args.train_last_n_layers)
        print(f"      froze {n_frozen} parameters; training the last "
              f"{args.train_last_n_layers} decoder layers")
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # With the first layers frozen, the activations entering the first trainable layer
        # carry no grad_fn, and reentrant checkpointing then drops the graph entirely --
        # loss.backward() fails with "element 0 of tensors does not require grad". Two
        # fixes are needed together: non-reentrant checkpointing, and forcing the input
        # embeddings to require grad so a graph exists from the start.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"      trainable: {n_train / 1e6:.0f} M parameters "
          f"(~{n_train * 12 / 1e9:.1f} GB for weights, grads and AdamW state)")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    step, t0, losses = 0, time.time(), []
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        optim.zero_grad(set_to_none=True)
        for n, i in enumerate(order, 1):
            out = model(**build_batch(train[i], processor, args.device))
            loss = out.loss / args.grad_accum
            loss.backward()
            losses.append(float(out.loss))
            if n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 4 == 0:
                    recent = float(np.mean(losses[-4 * args.grad_accum:]))
                    print(f"      epoch {epoch} step {step:4d}  loss {recent:.4f}  "
                          f"{time.time() - t0:.0f}s", flush=True)

    print(f"      trained {step} optimizer steps, final loss "
          f"{float(np.mean(losses[-8:])):.4f}")

    print(f"[5/5] Exporting to {args.output_path}")
    model.eval()
    export_quantized_model(model, tokenizer, processor,
                           model_dir=args.model_path, output_dir=args.output_path)
    print(f"Saved to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
