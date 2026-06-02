# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""
ASR Server — TensorRT-Edge-LLM pybind mode.

Engine loads ONCE at startup; requests are processed via an asyncio queue
(vLLM-style single-worker pattern).

OpenAI Whisper-compatible endpoint:
    POST /v1/audio/transcriptions

Configuration via environment variables:
    ENGINE_DIR        Path to the LLM engine directory
    AUDIO_ENGINE_DIR  Path to the audio encoder directory
                      (default: ENGINE_DIR/audio_encoder)
    EDGELLM_ROOT      TensorRT-Edge-LLM repository root
    TEMP_DIR          Temporary working directory (default: /tmp/asr_server)
    WAVES_DIR         Optional: path to a .wav file tree used by the file
                      browser endpoints (GET /v1/files, GET /v1/audio/file).
                      If unset, those endpoints return 404.

Usage:
    ENGINE_DIR=/path/to/Qwen3-ASR-Engines \\
    EDGELLM_ROOT=/path/to/TensorRT-Edge-LLM \\
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT-Edge-LLM/build \\
    uvicorn asr_server:app --host 0.0.0.0 --port 8000
"""

import _edgellm_runtime as rt
import asyncio
import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("asr_server")

# ---------------------------------------------------------------------------
# Supported language names (model outputs one of these as a prefix)
# ---------------------------------------------------------------------------

_SUPPORTED_LANGUAGES = {
    "chinese", "english", "cantonese", "arabic", "german", "french",
    "spanish", "portuguese", "indonesian", "italian", "korean", "russian",
    "thai", "vietnamese", "japanese", "turkish", "hindi", "malay",
    "dutch", "swedish", "danish", "finnish", "polish", "czech",
    "filipino", "persian", "greek", "romanian", "hungarian", "macedonian",
}

# Pattern: "language <LangName>" at the start of output (case-insensitive)
_LANG_PREFIX_RE = re.compile(
    r'^language\s+(' + '|'.join(_SUPPORTED_LANGUAGES) + r')\s*',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

ENGINE_DIR       = os.environ["ENGINE_DIR"]
AUDIO_ENGINE_DIR = os.environ.get("AUDIO_ENGINE_DIR",
                                  str(Path(ENGINE_DIR) / "audio_encoder"))
EDGELLM_ROOT     = os.environ["EDGELLM_ROOT"]
TEMP_DIR         = os.environ.get("TEMP_DIR", "/tmp/asr_server")
WAVES_DIR        = os.environ.get("WAVES_DIR", "")

PYBIND_SO_DIR = str(Path(EDGELLM_ROOT) / "build" / "pybind")
PLUGIN_LIB    = str(Path(EDGELLM_ROOT) / "build" / "libNvInfer_edgellm_plugin.so")

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

for _p, _name in [
    (ENGINE_DIR,       "ENGINE_DIR"),
    (AUDIO_ENGINE_DIR, "AUDIO_ENGINE_DIR"),
    (EDGELLM_ROOT,     "EDGELLM_ROOT"),
    (PYBIND_SO_DIR,    "PYBIND_SO_DIR"),
    (PLUGIN_LIB,       "PLUGIN_LIB"),
]:
    if not Path(_p).exists():
        raise RuntimeError(f"{_name} not found: {_p}")

log.info("ENGINE_DIR       : %s", ENGINE_DIR)
log.info("AUDIO_ENGINE_DIR : %s", AUDIO_ENGINE_DIR)
log.info("EDGELLM_ROOT     : %s", EDGELLM_ROOT)

# ---------------------------------------------------------------------------
# Load pybind runtime
# ---------------------------------------------------------------------------

os.environ["EDGELLM_PLUGIN_PATH"] = PLUGIN_LIB
if PYBIND_SO_DIR not in sys.path:
    sys.path.insert(0, PYBIND_SO_DIR)


log.info("Loading engines (this may take a moment)...")
_runtime = rt.LLMRuntime(ENGINE_DIR, AUDIO_ENGINE_DIR, {})
_runtime.capture_decoding_cuda_graph()
log.info("Engine loaded and ready.")

# ---------------------------------------------------------------------------
# In-process audio preprocessing
#
# Import preprocess_single_audio once at startup instead of spawning a new
# Python subprocess per request.  This eliminates ~300-800 ms of interpreter
# cold-start overhead on every call; only the mel-spectrogram computation
# itself is timed.  Falls back to subprocess mode if the import fails.
# ---------------------------------------------------------------------------

_edgellm_pkg_root = str(Path(EDGELLM_ROOT))
if _edgellm_pkg_root not in sys.path:
    sys.path.insert(0, _edgellm_pkg_root)

try:
    from tensorrt_edgellm.scripts.preprocess_audio import preprocess_single_audio
    log.info("Preprocess module imported successfully (in-process mode).")
    _PREPROCESS_INPROCESS = True
except ImportError as _e:
    log.warning("Cannot import preprocess module in-process (%s). "
                "Falling back to subprocess mode.", _e)
    import subprocess
    _PREPROCESS_INPROCESS = False


def _run_preprocess(audio_path: Path, safetensor_path: Path) -> None:
    """Convert an audio file to a SafeTensors mel-spectrogram.

    Uses the imported preprocess_single_audio when available (fast path),
    otherwise falls back to spawning a subprocess (slow path).
    """
    if _PREPROCESS_INPROCESS:
        preprocess_single_audio(str(audio_path), str(safetensor_path))
    else:
        proc = subprocess.run([
            "python", "-m", "tensorrt_edgellm.scripts.preprocess_audio",
            "--input",  str(audio_path),
            "--output", str(safetensor_path),
        ], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"preprocess failed: {proc.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Asyncio queue — single worker, requests processed sequentially
# ---------------------------------------------------------------------------

_queue: asyncio.Queue = None


async def _worker():
    while True:
        safetensor_path, temperature, future = await _queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _infer, safetensor_path, temperature)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            _queue.task_done()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ASR Server", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _queue
    _queue = asyncio.Queue()
    asyncio.create_task(_worker())
    log.info("Worker started.")


@app.get("/health")
def health():
    return {"status": "ok", "engine_dir": ENGINE_DIR}


# ---------------------------------------------------------------------------
# File browser endpoints (optional — requires WAVES_DIR env var)
# ---------------------------------------------------------------------------

@app.get("/v1/files")
def list_files():
    """Return .wav files grouped by speaker sub-directory.

    Requires the WAVES_DIR environment variable to point to the audio tree:
        WAVES_DIR/
            speaker_a/
                utt_001.wav
            speaker_b/
                ...
    """
    if not WAVES_DIR or not Path(WAVES_DIR).exists():
        raise HTTPException(status_code=404,
                            detail="WAVES_DIR not set or not found")

    result: dict = {}
    for speaker_dir in sorted(Path(WAVES_DIR).iterdir()):
        if not speaker_dir.is_dir():
            continue
        files = sorted(f.name for f in speaker_dir.glob("*.wav"))
        if files:
            result[speaker_dir.name] = files

    return {"speakers": result}


@app.get("/v1/audio/file")
def serve_audio_file(speaker: str, name: str):
    """Serve a single .wav file so the browser can fetch and POST it.

    GET /v1/audio/file?speaker=<speaker>&name=<filename.wav>
    """
    if not WAVES_DIR or not Path(WAVES_DIR).exists():
        raise HTTPException(status_code=404, detail="WAVES_DIR not set or not found")

    file_path = Path(WAVES_DIR) / speaker / name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if file_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return FileResponse(str(file_path), media_type="audio/wav", filename=name)


# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    temperature: float = Form(default=1.0),
    language: str = Form(default=""),
):
    """OpenAI Whisper-compatible transcription endpoint.

    Accepted formats: .wav  .mp3  .flac  .ogg  .m4a
    Returns: {"text": "<transcript>", "timings": {...}}
    """
    req_id = uuid.uuid4().hex[:8]
    work = Path(TEMP_DIR) / req_id
    work.mkdir(parents=True, exist_ok=True)

    suffix          = Path(file.filename).suffix if file.filename else ".wav"
    audio_path      = work / f"audio{suffix}"
    safetensor_path = work / "audio.safetensors"

    try:
        audio_path.write_bytes(await file.read())

        # Stage 1: Preprocess (mel-spectrogram extraction)
        t_pre_start = time.perf_counter()
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, _run_preprocess, audio_path, safetensor_path)
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail=f"preprocess failed: {exc}")
        preprocess_ms = (time.perf_counter() - t_pre_start) * 1000

        # Stage 2: LLM inference (includes asyncio queue wait, ~0 ms at batch=1)
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await _queue.put((safetensor_path, temperature, future))
        t_inf_start = time.perf_counter()
        transcript = await future
        inference_ms = (time.perf_counter() - t_inf_start) * 1000

        return {
            "text": transcript,
            "timings": {
                "preprocess_ms": round(preprocess_ms, 1),
                "inference_ms":  round(inference_ms, 1),
                "total_ms":      round(preprocess_ms + inference_ms, 1),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Request %s failed", req_id)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Strip the language-prefix token emitted by the model."""
    text = text.strip()
    text = _LANG_PREFIX_RE.sub("", text).strip()
    if "<asr_text>" in text:
        text = text.split("<asr_text>")[-1].strip()
    return text


def _infer(safetensor_path: Path, temperature: float) -> str:
    msg_system = rt.Message()
    msg_system.role = "system"
    c_sys = rt.MessageContent()
    c_sys.type = "text"
    c_sys.content = ""
    msg_system.contents = [c_sys]

    msg_user = rt.Message()
    msg_user.role = "user"
    c_audio = rt.MessageContent()
    c_audio.type = "audio"
    c_audio.content = str(safetensor_path)
    msg_user.contents = [c_audio]

    audio_data = rt.AudioData()
    audio_data.mel_spectrogram_path = str(safetensor_path)
    audio_data.mel_spectrogram_format = "safetensors"

    req = rt.Request(messages=[msg_system, msg_user])
    req.audio_buffers = [audio_data]

    gen_req = rt.LLMGenerationRequest()
    gen_req.requests = [req]
    gen_req.temperature = temperature
    gen_req.top_p = 1.0
    gen_req.top_k = 50
    gen_req.max_generate_length = 256
    gen_req.apply_chat_template = True
    gen_req.add_generation_prompt = True

    response = _runtime.handle_request(gen_req)
    text = response.output_texts[0] if response.output_texts else ""
    log.info("RAW OUTPUT: %r", text)
    return _normalize(text)
