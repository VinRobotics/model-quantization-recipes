# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
benchmark.py — Batch inference benchmark for Qwen3-ASR on Jetson Orin Nano
               using TensorRT-Edge-LLM engines.

Pipeline:
  Phase 1 — Preprocess audio files -> SafeTensor files  (CPU-bound, ~6-core Nano)
  Phase 2 — Build batch input JSON
  Phase 3 — Run llm_inference binary (loads engine once for all requests)
  Phase 4 — Compute WER / RTF / Throughput

Requirements (Jetson):
  pip install soundfile tqdm jiwer safetensors

Usage:
  python benchmark.py \
      --audio_dir   /path/to/wav/files \
      --prompts     /path/to/prompts.txt \
      --engine_dir  /path/to/Engines \
      --work_dir    ./benchmark_output \
      [--trt_edgellm_dir ~/TensorRT-Edge-LLM] \
      [--workers 6]

prompts.txt format (one utterance per line):
  <utterance_id> <reference transcript>
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from jiwer import wer as compute_wer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir",       required=True)
    p.add_argument("--prompts",         required=True)
    p.add_argument("--engine_dir",      required=True)
    p.add_argument("--work_dir",        default="./benchmark_output")
    p.add_argument("--trt_edgellm_dir", default=os.path.expanduser("~/TensorRT-Edge-LLM"))
    p.add_argument("--workers",         type=int, default=None)
    p.add_argument("--max_generate_length", type=int, default=256)
    p.add_argument("--temperature",     type=float, default=0.0)
    p.add_argument("--top_p",           type=float, default=1.0)
    p.add_argument("--top_k",           type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument(
        "--language",
        default="",
        help="Language for WER normalisation e.g. chinese, vietnamese. Character-based scripts use CER.")
    return p.parse_args()


# Languages whose orthography is character-based (no word-boundary spaces).
# For these, we insert spaces between characters so jiwer computes CER-style WER.
_CHAR_BASED_LANGS: frozenset = frozenset({"chinese", "cantonese", "japanese"})


def normalize(text: str, language: str = "") -> str:
    """Normalise model output for WER/CER computation.

    Works for all Qwen3-ASR supported languages. Key steps:
      1. Strip the 'language <Name>' prefix the model may prepend (any language).
      2. Extract text from <asr_text> tags if present.
      3. Lowercase.
      4. For character-based scripts (Chinese, Cantonese, Japanese): remove
         punctuation while preserving CJK codepoints, then insert spaces between
         characters to enable character-level error rate via jiwer.
      5. For all other languages: remove non-word, non-space characters.
    """
    text = text.strip()
    # Strip 'language <Name>' prefix produced by the model (generalised, any language)
    text = re.sub(r"^language\s+\w+", "", text, flags=re.IGNORECASE).strip()
    # Extract from <asr_text> tags
    if "<asr_text>" in text:
        text = text.split("<asr_text>")[-1].strip()
    text = text.lower()
    lang = language.lower()
    if lang in _CHAR_BASED_LANGS:
        # Keep CJK codepoints (CJK Unified, Hiragana, Katakana, Hangul),
        # remove punctuation and other non-alphanumeric chars
        text = re.sub(
            r"[^一-鿿぀-ヿ가-힯\w\s]", "", text
        )
        # Insert spaces between CJK characters for character-level WER via jiwer
        text = re.sub(r"([一-鿿぀-ヿ가-힯])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        # Alphabetic / syllabic scripts (Latin, Arabic, Cyrillic, Thai, Devanagari …)
        # Remove all punctuation; keep word characters and spaces.
        text = re.sub(r"[^\w\s]", "", text)
    return text


def get_duration_s(wav_path: str) -> float:
    import soundfile as sf
    return sf.info(wav_path).duration


def preprocess_audio(wav_path: str, out_path: str) -> None:
    if os.path.exists(out_path):
        return
    subprocess.run(
        ["python", "-m", "tensorrt_edgellm.scripts.preprocess_audio",
         "--input", wav_path, "--output", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def parse_tegrastats_ram_mb(log_path: str) -> int:
    peak = 0
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"RAM (\d+)/\d+MB", line)
                if m:
                    peak = max(peak, int(m.group(1)))
    except FileNotFoundError:
        pass
    return peak


def parse_inference_time_s(log_path: str) -> float:
    fmt = r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\]"
    t0 = t1 = None
    with open(log_path) as f:
        for line in f:
            if t0 is None and "Processing" in line and "batched requests" in line \
                    and "Progress" not in line:
                m = re.search(fmt, line)
                if m:
                    t0 = m.group(1)
            if "Processing complete" in line and re.search(r"\d+/\d+", line):
                m = re.search(fmt, line)
                if m:
                    t1 = m.group(1)
    if not t0 or not t1:
        raise RuntimeError(f"Could not parse inference timestamps from {log_path}.")
    fmt_s = "%H:%M:%S.%f"
    delta = (datetime.strptime(t1, fmt_s) - datetime.strptime(t0, fmt_s)).total_seconds()
    return delta + 86400 if delta < 0 else delta


def main():
    args      = parse_args()
    n_workers = args.workers or os.cpu_count()

    work_dir       = Path(args.work_dir)
    safetensor_dir = work_dir / "safetensors"
    results_dir    = work_dir / "results"
    input_json     = work_dir / "input_batch.json"
    output_json    = work_dir / "output_batch.json"
    llm_log        = work_dir / "llm_inference.log"
    tegra_log      = work_dir / "tegrastats.log"

    for d in [safetensor_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    llm_bin = Path(args.trt_edgellm_dir) / "build/examples/llm/llm_inference"
    if not llm_bin.exists():
        raise FileNotFoundError(f"llm_inference binary not found: {llm_bin}")

    wav_files = sorted(Path(args.audio_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {args.audio_dir}")
    print(f"[INFO] Found {len(wav_files)} audio files.")

    print(f"\n=== Phase 1: Preprocessing audio ({n_workers} workers) ===")
    print("Note: preprocessing is CPU-bound. On Jetson Orin Nano (6-core) "
          "this may take several minutes for large sets.")

    tasks = [
        (str(w), str(safetensor_dir / (w.stem + ".safetensors")))
        for w in wav_files
    ]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(preprocess_audio, src, dst): src for src, dst in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="  preprocessing", unit="file", ncols=70):
            fut.result()

    durations = {wav.stem: get_duration_s(str(wav)) for wav in wav_files}
    total_audio_duration = sum(durations.values())

    print("\n=== Phase 2: Building batch input JSON ===")
    safetensor_files = sorted(safetensor_dir.glob("*.safetensors"))
    requests_payload = [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": str(sf)}]},
            ]
        }
        for sf in safetensor_files
    ]
    payload = {
        "batch_size":          args.batch_size,
        "temperature":         args.temperature,
        "top_p":               args.top_p,
        "top_k":               args.top_k,
        "max_generate_length": args.max_generate_length,
        "requests":            requests_payload,
    }
    with open(input_json, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Batch JSON: {len(requests_payload)} requests -> {input_json}")

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        tmp_path = tmp.name
    proc = subprocess.Popen(["tegrastats", "--interval", "200"],
                            stdout=open(tmp_path, "w"), stderr=subprocess.DEVNULL)
    time.sleep(0.8)
    proc.terminate()
    proc.wait()
    ram_baseline_mb = parse_tegrastats_ram_mb(tmp_path)
    os.unlink(tmp_path)
    print(f"[INFO] RAM baseline: {ram_baseline_mb} MB")

    print("\n=== Phase 3: Running inference ===")
    os.environ.setdefault("LD_LIBRARY_PATH",
                          str(Path(args.trt_edgellm_dir) / "build"))

    tegra_proc = subprocess.Popen(
        ["tegrastats", "--interval", "200"],
        stdout=open(tegra_log, "w"), stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)

    with open(llm_log, "w") as log_f:
        subprocess.run(
            [
                str(llm_bin),
                "--engineDir",          args.engine_dir,
                "--multimodalEngineDir", str(Path(args.engine_dir) / "audio_encoder"),
                "--inputFile",          str(input_json),
                "--outputFile",         str(output_json),
            ],
            check=True, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=args.trt_edgellm_dir,
        )

    tegra_proc.terminate()
    tegra_proc.wait()

    ram_peak_mb  = parse_tegrastats_ram_mb(str(tegra_log))
    ram_delta_mb = ram_peak_mb - ram_baseline_mb
    infer_time_s = parse_inference_time_s(str(llm_log))

    print(f"[INFO] Inference time (decode only): {infer_time_s:.4f}s")
    print(f"[INFO] RAM footprint (peak - baseline): {ram_delta_mb} MB")

    print("\n=== Phase 4: Computing metrics ===")

    refs: dict = {}
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                refs[parts[0]] = normalize(parts[1], language=args.language)

    with open(output_json, encoding="utf-8") as f:
        output = json.load(f)
    responses = output if isinstance(output, list) else output.get("responses", output.get("results", []))

    ids = sorted(p.stem for p in safetensor_files)

    references, hypotheses, detail_lines = [], [], []
    missing_refs = 0
    for uid, resp in tqdm(zip(ids, responses), total=len(ids),
                          desc="  evaluating", unit="sample", ncols=70):
        hyp_raw = resp.get("output_text", "") if isinstance(resp, dict) else str(resp)
        hyp = normalize(hyp_raw, language=args.language)
        if uid not in refs:
            missing_refs += 1
            continue
        references.append(refs[uid])
        hypotheses.append(hyp)
        detail_lines.append(f"{uid}\tREF: {refs[uid]}\tHYP: {hyp}")

    n_eval         = len(references)
    overall_wer    = compute_wer(references, hypotheses)
    rtf            = infer_time_s / total_audio_duration if total_audio_duration > 0 else 0.0
    throughput_sps = n_eval / infer_time_s if infer_time_s > 0 else 0.0
    throughput_rt  = total_audio_duration / infer_time_s if infer_time_s > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  Samples evaluated : {n_eval}")
    print(f"  Missing refs      : {missing_refs}")
    print(f"  WER               : {overall_wer:.4f}  ({overall_wer * 100:.2f}%)")
    print(f"  Total audio dur   : {total_audio_duration:.2f}s")
    print(f"  Inference time    : {infer_time_s:.4f}s")
    print(f"  RTF               : {rtf:.4f}  (< 1.0 = faster than real-time)")
    print(f"  Throughput        : {throughput_sps:.2f} samples/s")
    print(f"                      {throughput_rt:.2f}x real-time")
    print(f"  RAM footprint     : {ram_delta_mb} MB")
    print("=" * 60)

    detail_path  = results_dir / "detail.tsv"
    summary_path = results_dir / "summary.txt"

    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("ID\tREF\tHYP\n")
        for line in detail_lines:
            f.write(line + "\n")

    with open(summary_path, "w") as f:
        f.write(f"samples_evaluated={n_eval}\n")
        f.write(f"wer={overall_wer:.6f}\n")
        f.write(f"total_audio_duration_s={total_audio_duration:.4f}\n")
        f.write(f"inference_time_s={infer_time_s:.6f}\n")
        f.write(f"rtf={rtf:.6f}\n")
        f.write(f"throughput_samples_per_sec={throughput_sps:.5f}\n")
        f.write(f"throughput_realtime_factor={throughput_rt:.4f}\n")
        f.write(f"ram_footprint_mb={ram_delta_mb}\n")

    print(f"\n  Detail log -> {detail_path}")
    print(f"  Summary    -> {summary_path}")


if __name__ == "__main__":
    main()
