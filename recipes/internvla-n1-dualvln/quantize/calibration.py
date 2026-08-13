# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""Calibration dataloaders for Qwen2.5-VL quantization.

Three dataloader paths:
  * ``text_calib_dataloader`` — ``cnn_dailymail`` ``input_ids`` only, used when
    quantizing LLM backbone alone (no visual quantization).
  * ``multimodal_calib_dataloader`` — ``lmms-lab/MMMU`` image+text pairs streamed
    through the model's own ``AutoProcessor`` chat template, used when visual
    tower quantization is enabled.
  * ``vln_calib_dataloader`` — the in-distribution set: real InternData-N1 VLN-CE
    episodes (the data System 2 was fine-tuned on) assembled through the deployed
    agent's own prompt path, so ModelOpt sees the true activation ranges of the
    farthest-pixel-goal navigation task instead of out-of-distribution web text/QA.

The first two mirror NVIDIA's reference implementation.
"""

import os
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Offline fallbacks
# --------------------------------------------------------------------------- #
def _hub_reachable(timeout: float = 5.0) -> bool:
    """Cheap reachability probe for huggingface.co.

    Returns False on a TLS interception proxy as well as on a real outage, which is
    what we want: in both cases the cached parquet is the usable path.
    """
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co/api/whoami-v2", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # reachable, just unauthenticated
    except Exception:
        return False


def _local_parquet_split(dataset_name: str, split: str) -> list[str]:
    """Find cached parquet shards for ``dataset_name``/``split`` in the HF hub cache."""
    cache = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache, "datasets--" + dataset_name.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return []
    found: list[str] = []
    for root, _dirs, files in os.walk(snapshots):
        for name in sorted(files):
            if name.startswith(split) and name.endswith(".parquet"):
                found.append(os.path.join(root, name))
    return sorted(found)


# --------------------------------------------------------------------------- #
# Text calibration (LLM-only strategies S1, S2)
# --------------------------------------------------------------------------- #
def text_calib_dataloader(
    tokenizer,
    dataset_name: str = "abisee/cnn_dailymail",
    batch_size: int = 1,
    num_samples: int = 512,
    max_length: int = 512,
) -> DataLoader:
    """Return a DataLoader of tokenised ``input_ids`` for calibration.

    Mirrors NVIDIA's ``_text_calib_dataloader``. ``max_length`` only governs
    tokenizer truncation during this call and does NOT modify
    ``tokenizer.model_max_length`` — verified by an assert in the caller.
    """
    from datasets import load_dataset

    # A Jetson is often behind a TLS-intercepting proxy or fully air-gapped, where
    # load_dataset() fails even though the parquet shards are already in the hub cache
    # (HF's own offline mode does not help: it still wants the dataset script). Fall
    # back to reading those shards directly so a calibration run does not depend on
    # network reachability. Set HF_DATASETS_LOCAL_ONLY=1 to skip the hub attempt.
    local = _local_parquet_split(dataset_name, split="train")
    if local and (os.environ.get("HF_DATASETS_LOCAL_ONLY") == "1" or not _hub_reachable()):
        print(f"      [calib] reading cached parquet ({len(local)} shard(s)) instead of the Hub")
        ds = load_dataset("parquet", data_files=local, split="train")
        col = "article" if "article" in ds.column_names else ds.column_names[0]
        texts = ds[col][:num_samples]
    elif "abisee/cnn_dailymail" in dataset_name:
        ds = load_dataset(dataset_name, name="3.0.0", split="train")
        texts = ds["article"][:num_samples]
    else:
        ds = load_dataset(dataset_name, split="train")
        if "text" in ds.column_names:
            col = "text"
        elif "article" in ds.column_names:
            col = "article"
        else:
            raise ValueError(
                f"Dataset {dataset_name!r} has no 'text' or 'article' column: "
                f"{ds.column_names}"
            )
        texts = ds[col][:num_samples]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------- #
# Multimodal calibration (visual-quantization strategies S3, S4)
# --------------------------------------------------------------------------- #
def _iter_image_question_pairs(dataset_name: str):
    """Yield ``(image, question)`` pairs from a HuggingFace calibration dataset.

    Mirrors NVIDIA's ``_iter_image_question_pairs``. Tolerant of two common
    schemas:
      * ScienceQA-style: single ``image`` column.
      * MMMU-style: numbered ``image_1`` / ``image_2`` / ... columns.

    Splits are tried in the order ``dev`` → ``validation`` → ``train``.
    """
    from datasets import load_dataset

    last_err: Optional[Exception] = None
    ds = None
    for split in ("dev", "validation", "train"):
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(
            f"Could not load {dataset_name!r} via any of "
            f"split=dev/validation/train"
        ) from last_err

    for example in ds:
        image = example.get("image")
        if image is None:
            for i in range(1, 8):
                image = example.get(f"image_{i}")
                if image is not None:
                    break
        question = example.get("question") or ""
        if image is not None and question:
            yield image, question


def multimodal_calib_dataloader(
    processor,
    dataset_name: str = "lmms-lab/MMMU",
    num_samples: int = 128,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Materialise a list of ``BatchFeature`` dicts with ``input_ids`` + ``pixel_values``.

    Mirrors NVIDIA's ``_multimodal_calib_dataloader``. Streams image-question
    pairs through the model's own ``AutoProcessor`` chat template so the visual
    tower receives real activations.

    NVIDIA caps multimodal calibration at 128 samples because VLM calibration
    is GPU-memory bound. The caller is expected to enforce this cap.

    Returns a *list* (not generator) so ModelOpt can re-iterate forward_loop
    during algorithm selection (mirrors NVIDIA's design).
    """
    batches: list[dict[str, Any]] = []
    for image, question in _iter_image_question_pairs(dataset_name):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        batches.append({
            k: v
            for k, v in inputs.items()
            if v is not None
            and not (isinstance(v, torch.Tensor) and v.numel() == 0)
        })
        if len(batches) >= num_samples:
            break

    if not batches:
        raise RuntimeError(
            f"No usable multimodal samples from {dataset_name!r}. "
            "Check dataset access / processor chat template."
        )
    return batches


# --------------------------------------------------------------------------- #
# VLN calibration (in-distribution — InternData-N1 VLN-CE)
# --------------------------------------------------------------------------- #
def _import_prompt_builder():
    """Import ``lib/prompt_builder`` (the single source of truth for the System 2
    prompt) regardless of how this module was launched. Kept lazy so the text/MMMU
    paths never pay for it."""
    import os
    import sys

    lib = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "lib",
    )
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import prompt_builder  # noqa: E402

    return prompt_builder


def _discover_vln_episodes(data_root: str, rgb_key: str):
    """Walk a LeRobot dataset tree and yield one record per episode.

    ``data_root`` may be a single scene dir or any parent of several — every dir
    holding ``meta/episodes.jsonl`` is picked up. Frames are stored as per-frame
    JPGs under ``videos/chunk-XXX/<rgb_key>/episode_{idx:06d}_{frame}.jpg`` (no
    video decoding needed). Falls back to the first available ``*.rgb.*`` stream
    when ``rgb_key`` is absent in a scene.
    """
    import glob
    import json
    import os

    records = []
    meta_files = glob.glob(
        os.path.join(data_root, "**", "meta", "episodes.jsonl"), recursive=True
    )
    if not os.path.isdir(os.path.join(data_root, "meta")) and not meta_files:
        raise FileNotFoundError(
            f"No LeRobot episodes.jsonl found under {data_root!r} "
            "(expected <scene>/meta/episodes.jsonl)."
        )
    # Include data_root itself if it is a scene dir.
    if os.path.isfile(os.path.join(data_root, "meta", "episodes.jsonl")):
        meta_files.append(os.path.join(data_root, "meta", "episodes.jsonl"))

    for meta_file in sorted(set(meta_files)):
        scene_dir = os.path.dirname(os.path.dirname(meta_file))
        info_path = os.path.join(scene_dir, "meta", "info.json")
        chunks_size = 1000
        if os.path.isfile(info_path):
            chunks_size = json.load(open(info_path)).get("chunks_size", 1000)

        # Resolve the rgb stream dir for this scene (prefer the requested key).
        video_root = os.path.join(scene_dir, "videos")
        chunk_dirs = sorted(glob.glob(os.path.join(video_root, "chunk-*")))
        if not chunk_dirs:
            continue

        def _rgb_dir_for_chunk(chunk_dir):
            want = os.path.join(chunk_dir, rgb_key)
            if os.path.isdir(want):
                return want
            alts = sorted(glob.glob(os.path.join(chunk_dir, "*.rgb.*")))
            return alts[0] if alts else None

        with open(meta_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep_idx = ep["episode_index"]
                length = ep.get("length", 0)
                tasks = ep.get("tasks") or []
                if length <= 0 or not tasks:
                    continue
                chunk_dir = os.path.join(
                    video_root, f"chunk-{ep_idx // chunks_size:03d}"
                )
                rgb_dir = _rgb_dir_for_chunk(chunk_dir) or _rgb_dir_for_chunk(
                    chunk_dirs[0]
                )
                if rgb_dir is None:
                    continue
                records.append(
                    {
                        "scene_dir": scene_dir,
                        "ep_idx": ep_idx,
                        "length": length,
                        "instruction": tasks[0],
                        "rgb_dir": rgb_dir,
                    }
                )
    return records


def vln_calib_dataloader(
    processor,
    data_root: str,
    num_samples: int = 128,
    rgb_key: str = "observation.images.rgb.125cm_0deg",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Materialise calibration ``BatchFeature`` dicts from InternData-N1 VLN-CE.

    Each sample is one navigation step: a real instruction + a sequence of
    egocentric RGB frames (sub-sampled history + current), assembled through the
    SAME path the deployed agent uses (``prompt_builder.build_sample_inputs``), so
    the number of history frames, the prompt template and the ``<image>`` layout
    all match ``InternVLAN1Net.s2_step`` exactly.

    Returns a *list* (not a generator) so ModelOpt can re-iterate the forward loop
    during algorithm selection, mirroring ``multimodal_calib_dataloader``.
    """
    import os
    import random
    import shutil
    import tempfile

    import numpy as np
    from PIL import Image

    pb = _import_prompt_builder()
    num_history = pb.NUM_HISTORY  # single source of truth — must match the prompt

    def frame_path(rgb_dir, ep_idx, frame):
        return os.path.join(rgb_dir, f"episode_{ep_idx:06d}_{frame}.jpg")

    episodes = _discover_vln_episodes(data_root, rgb_key)
    if not episodes:
        raise RuntimeError(
            f"No usable VLN episodes under {data_root!r} (rgb_key={rgb_key!r})."
        )

    # The deployed agent resizes every RGB frame to (resize_w, resize_h) before the
    # processor (internvla_n1_policy.py). Match that here so the visual-token count
    # and ViT activations track deployment, not the raw camera resolution. Resized
    # frames are cached under a temp dir and cleaned up once every batch is built
    # (the returned batches hold materialised tensors, not paths).
    tmp_dir = tempfile.mkdtemp(prefix="vln_calib_")
    resized_cache: dict[str, str] = {}

    def resized_frame(src_path):
        cached = resized_cache.get(src_path)
        if cached is not None:
            return cached
        img = Image.open(src_path).convert("RGB").resize(
            (pb.RESIZE_W, pb.RESIZE_H)
        )
        dst = os.path.join(tmp_dir, f"f{len(resized_cache):06d}.jpg")
        img.save(dst, quality=95)
        resized_cache[src_path] = dst
        return dst

    rng = random.Random(seed)
    batches: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = num_samples * 20

    try:
        while len(batches) < num_samples and attempts < max_attempts:
            attempts += 1
            ep = rng.choice(episodes)
            length = ep["length"]
            # Pick a "current" step; prefer t>=1 so the sample carries history.
            t = rng.randint(1, length - 1) if length > 1 else 0

            # History frame indices — identical rule to
            # prompt_builder.build_conversation (same NUM_HISTORY constant).
            if t == 0:
                hist = []
            else:
                hist = np.unique(
                    np.linspace(0, t - 1, num_history, dtype=np.int32)
                ).tolist()

            src_paths = [frame_path(ep["rgb_dir"], ep["ep_idx"], h) for h in hist]
            src_paths.append(frame_path(ep["rgb_dir"], ep["ep_idx"], t))
            if not all(os.path.isfile(p) for p in src_paths):
                continue  # frame missing on disk — skip this sample

            sample = {
                "images": [resized_frame(p) for p in src_paths],
                "episode_idx": t,
                "instruction": ep["instruction"],
            }
            try:
                inputs = pb.build_sample_inputs(sample, processor)
            except Exception:  # noqa: BLE001 — drop malformed samples, keep going
                continue

            batches.append(
                {
                    k: v
                    for k, v in inputs.items()
                    if v is not None
                    and not (isinstance(v, torch.Tensor) and v.numel() == 0)
                }
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not batches:
        raise RuntimeError(
            f"Could not build any VLN calibration sample from {data_root!r} "
            f"after {attempts} attempts (rgb_key={rgb_key!r})."
        )
    return batches