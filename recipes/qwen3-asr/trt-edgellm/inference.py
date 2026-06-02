# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
inference.py — Single-audio inference using a TensorRT-Edge-LLM engine.

Preprocesses one audio file, runs the LLM inference binary,
and prints only the transcription text.

Requirements (Jetson):
  pip install soundfile safetensors

Usage:
  python inference.py \
      --audio      /path/to/audio.wav \
      --engine_dir /path/to/Engines \
      [--trt_edgellm_dir ~/TensorRT-Edge-LLM] \
      [--max_generate_length 256]
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio",           required=True,
                   help="Path to input .wav file")
    p.add_argument("--engine_dir",      required=True,
                   help="Path to TRT engine directory (must contain audio_encoder/)")
    p.add_argument("--trt_edgellm_dir", default=os.path.expanduser("~/TensorRT-Edge-LLM"),
                   help="Path to TensorRT-Edge-LLM repo (default: ~/TensorRT-Edge-LLM)")
    p.add_argument("--max_generate_length", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.audio).resolve()
    engine_dir = Path(args.engine_dir).resolve()
    llm_bin    = Path(args.trt_edgellm_dir) / "build/examples/llm/llm_inference"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not llm_bin.exists():
        raise FileNotFoundError(f"llm_inference binary not found: {llm_bin}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir          = Path(tmpdir)
        safetensor_path = tmpdir / (audio_path.stem + ".safetensors")
        input_json      = tmpdir / "input.json"
        output_json     = tmpdir / "output.json"

        subprocess.run(
            ["python", "-m", "tensorrt_edgellm.scripts.preprocess_audio",
             "--input", str(audio_path), "--output", str(safetensor_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        payload = {
            "batch_size":          1,
            "temperature":         0.0,
            "top_p":               1.0,
            "top_k":               50,
            "max_generate_length": args.max_generate_length,
            "requests": [{
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user",   "content": [
                        {"type": "audio", "audio": str(safetensor_path)}
                    ]},
                ]
            }],
        }
        with open(input_json, "w") as f:
            json.dump(payload, f)

        os.environ.setdefault("LD_LIBRARY_PATH",
                              str(Path(args.trt_edgellm_dir) / "build"))
        subprocess.run(
            [
                str(llm_bin),
                "--engineDir",           str(engine_dir),
                "--multimodalEngineDir", str(engine_dir / "audio_encoder"),
                "--inputFile",           str(input_json),
                "--outputFile",          str(output_json),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=args.trt_edgellm_dir,
        )

        with open(output_json) as f:
            result = json.load(f)

        responses = (result if isinstance(result, list)
                     else result.get("responses", result.get("results", [])))
        if responses:
            resp = responses[0]
            text = resp.get("output_text", "") if isinstance(resp, dict) else str(resp)
            if "<asr_text>" in text:
                text = text.split("<asr_text>")[-1].strip()
            print(text.strip())


if __name__ == "__main__":
    main()
