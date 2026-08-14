#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
#
# Fetch a small, diverse subset of InternData-N1 VLN-CE scenes for calibration.
#
# InternData-N1 is gated on HuggingFace and the full VLN-CE traj_data is ~2.5 TB across 914
# scenes, so this pulls only a handful of per-scene archives. Accept the dataset terms and
# run `huggingface-cli login` first, or every download 401s.
#
# The default set spans r2r + rxr + scalevln -- the same training mix the base model saw --
# and is deliberately made of small scenes to keep disk in check. Override SCENES to pick
# others.
#
# Note one of these scenes, YmJkqBEsHnH, also appears in the held-out probe set used by
# quantize/benchmark_accuracy.py. Exclude it from either side before reading a number as
# out-of-sample; qat.py already does.

set -euo pipefail

OUTPUT_PATH="${CALIB_DATA_ROOT:-$HOME/vln-opt-work/calib_scenes}"
SCENES="${SCENES:-\
vln_ce/traj_data/r2r/gZ6f7yhEvPG.tar.gz \
vln_ce/traj_data/r2r/YmJkqBEsHnH.tar.gz \
vln_ce/traj_data/r2r/XcA2TqTSSAj.tar.gz \
vln_ce/traj_data/rxr/Pm6F8kyY3z2.tar.gz \
vln_ce/traj_data/rxr/PuKPg4mmafe.tar.gz \
vln_ce/traj_data/scalevln/00493-pUneSGJDrvY.tar.gz \
vln_ce/traj_data/scalevln/00446-tL6i2PtktSh.tar.gz \
vln_ce/traj_data/scalevln/00351-QxfX5te1gFu.tar.gz \
vln_ce/traj_data/scalevln/00335-janiYDpzM9j.tar.gz}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_path) OUTPUT_PATH="$2"; shift 2 ;;
        --scenes)      SCENES="$2"; shift 2 ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTPUT_PATH"
export INTERNDATA_DEST="$OUTPUT_PATH" INTERNDATA_SCENES="$SCENES"

python - <<'PY'
import os
import shutil
import tarfile
import tempfile

from huggingface_hub import hf_hub_download

REPO = "InternRobotics/InternData-N1"
dest = os.environ["INTERNDATA_DEST"]
scenes = os.environ["INTERNDATA_SCENES"].split()

# Download into a scratch dir and delete each archive right after extraction, so the HF blob
# cache never accumulates a second copy. local_dir avoids the shared cache entirely.
scratch = tempfile.mkdtemp(prefix="interndata_")
try:
    for rel in scenes:
        scene = os.path.basename(rel)[:-len(".tar.gz")]
        subset = rel.split("/")[2]                       # r2r | rxr | scalevln
        out_dir = os.path.join(dest, subset, scene)
        if os.path.isdir(os.path.join(out_dir, "meta")):
            print(f"[skip] already extracted: {out_dir}")
            continue
        print(f"[get ] {rel}")
        tar_path = hf_hub_download(REPO, rel, repo_type="dataset",
                                   local_dir=scratch, local_dir_use_symlinks=False)
        with tarfile.open(tar_path) as tf:
            tf.extractall(os.path.join(dest, subset))
        os.remove(tar_path)
        print(f"[ok  ] {scene}")
finally:
    shutil.rmtree(scratch, ignore_errors=True)

found = sum(1 for root, _, files in os.walk(dest)
            if root.endswith(os.sep + "meta") and "episodes.jsonl" in files)
print(f"[done] {found} scene(s) with episodes.jsonl under {dest}")
PY

echo "Calibration data ready: $OUTPUT_PATH"
echo "Pass it as --calib_data_root, or set CALIB_DATA_ROOT."
