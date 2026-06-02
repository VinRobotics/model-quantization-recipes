# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import time
import requests
import io
import re
import soundfile as sf
from datasets import load_dataset
from jiwer import wer as compute_wer
import datasets

datasets.config.AUDIO_DECODER = "soundfile"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url",   default="http://localhost:8000/v1/audio/transcriptions")
    p.add_argument("--model", required=True, help="Served model name or path")
    p.add_argument("--dataset", default="AILAB-VNUHCM/vivos",
                   help="HuggingFace dataset ID (default: AILAB-VNUHCM/vivos)")
    p.add_argument("--split", default="test", help="Dataset split (default: test)")
    return p.parse_args()


def normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def transcribe_http(url: str, model: str, audio, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    r = requests.post(
        url,
        files={"file": ("audio.wav", buf, "audio/wav")},
        data={"model": model},
        timeout=600,
    )
    text = r.json().get("text", "")
    if "<asr_text>" in text:
        text = text.split("<asr_text>")[-1].strip()
    return normalize(text)


def main():
    args = parse_args()

    ds      = load_dataset(args.dataset, split=args.split)
    samples = [{"audio": s["audio"], "text": s["sentence"]} for s in ds]
    n_eval  = len(samples) - 1

    print(f"[INFO] url={args.url}  model={args.model}  dataset={args.dataset}/{args.split}")
    print("[INFO] Warming up...")
    transcribe_http(
        args.url, args.model,
        samples[0]["audio"]["array"],
        samples[0]["audio"]["sampling_rate"],
    )
    print("[INFO] Warmup done.\n")

    hypotheses, references, latencies, durations = [], [], [], []

    for i, s in enumerate(samples[1:]):
        audio    = s["audio"]["array"]
        sr       = s["audio"]["sampling_rate"]
        ref      = normalize(s["text"])
        duration = len(audio) / sr
        start    = time.perf_counter()
        try:
            hyp     = transcribe_http(args.url, args.model, audio, sr)
            elapsed = time.perf_counter() - start
            hypotheses.append(hyp)
            references.append(ref)
            latencies.append(elapsed)
            durations.append(duration)
            print(
                f"[{i + 1}/{n_eval}] dur={duration:.1f}s  elapsed={elapsed:.2f}s  "
                f"RTF={elapsed / duration:.3f} | {hyp}"
            )
        except Exception as e:
            print(f"[{i + 1}/{n_eval}] SKIP — {e}")

    wer_score = compute_wer(references, hypotheses)
    print("\n=== RESULT ===")
    print(f"Completed  : {len(latencies)}/{n_eval}")
    print(f"WER        : {wer_score:.4f}  ({wer_score * 100:.2f}%)")
    print(f"Avg RTF    : {sum(lat / d for lat, d in zip(latencies, durations)) / len(latencies):.4f}")
    print(f"Overall RTF: {sum(latencies) / sum(durations):.4f}")
    print(f"Avg latency: {sum(latencies) / len(latencies):.2f}s")
    print(f"Throughput : {len(latencies) / sum(latencies):.2f} req/s")


if __name__ == "__main__":
    main()
