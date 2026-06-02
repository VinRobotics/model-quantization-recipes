#!/bin/bash
# init_submodules.sh — 1-click initialiser for third-party dependencies.
#
# Run once after cloning this repo:
#   bash external/init_submodules.sh
#
# What it sets up:
#   external/qwen3_asr/          — Qwen3-ASR repo (pip install -e .)
#   external/tensorrt_edge_llm/  — TensorRT-Edge-LLM 0.6.0 (build separately)
#
# NOTE: TensorRT-Edge-LLM must be compiled from source on Jetson Orin AGX.
#       See trt-edgellm/README.md for the build guide (including the
#       -DEMBEDDED_TARGET=jetson-orin CMake flag and the INT8 builder patch).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[init_submodules] Repo root: ${REPO_ROOT}"

# ── Qwen3-ASR ─────────────────────────────────────────────────────────────────
QWEN_DIR="${REPO_ROOT}/external/qwen3_asr"
if [ -d "${QWEN_DIR}/.git" ]; then
    echo "[OK] external/qwen3_asr already cloned — pulling latest"
    git -C "${QWEN_DIR}" pull --ff-only
else
    echo "[clone] Cloning QwenLM/Qwen3-ASR → external/qwen3_asr"
    git clone https://github.com/QwenLM/Qwen3-ASR.git "${QWEN_DIR}"
fi
echo "[pip] Installing Qwen3-ASR (editable)"
pip install -e "${QWEN_DIR}" --quiet

# ── TensorRT-Edge-LLM 0.6.0 ──────────────────────────────────────────────────
TRT_DIR="${REPO_ROOT}/external/tensorrt_edge_llm"
TRT_TAG="v0.6.0"
TRT_REPO="https://github.com/NVIDIA/TensorRT-Edge-LLM.git"

if [ -d "${TRT_DIR}/.git" ]; then
    echo "[OK] external/tensorrt_edge_llm already cloned"
else
    echo "[clone] Cloning TensorRT-Edge-LLM ${TRT_TAG} → external/tensorrt_edge_llm"
    git clone --branch "${TRT_TAG}" --depth 1 "${TRT_REPO}" "${TRT_DIR}"
fi
echo ""
echo "[IMPORTANT] TensorRT-Edge-LLM must be compiled from source on Jetson Orin AGX."
echo "            See trt-edgellm/README.md for the full build guide."
echo ""
echo "[done] init_submodules.sh completed."
