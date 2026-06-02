# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
import torch
import shutil
import glob
import os
import argparse
import sys
import modelopt.torch.opt as mto
from modelopt.torch.export import export_hf_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True, help="Path to Qwen3-ASR-1.7B")
    p.add_argument("--qwen_asr_root", required=True, help="Path to Qwen3-ASR repo root")
    p.add_argument("--quant_pt",      required=True, help="Path to quantized .pt checkpoint")
    p.add_argument("--tmp_dir",       required=True, help="Temporary dir for ModelOpt HF export")
    p.add_argument("--export_dir",    required=True, help="Output vLLM-compatible checkpoint dir")
    p.add_argument("--device",        default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.qwen_asr_root)

    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    print("[STEP 1] Exporting quantized weights via ModelOpt...")
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=args.device
    )
    mto.restore(model.thinker, args.quant_pt)
    with torch.inference_mode():
        export_hf_checkpoint(model.thinker, export_dir=args.tmp_dir)
    del model
    torch.cuda.empty_cache()
    print(f"[STEP 1] Done. Exported to {args.tmp_dir}")

    print("[STEP 2] Building vLLM-compatible checkpoint...")
    shutil.copytree(args.model_path, args.export_dir, dirs_exist_ok=True)
    shutil.copy(
        f"{args.tmp_dir}/hf_quant_config.json",
        f"{args.export_dir}/hf_quant_config.json",
    )
    for f in glob.glob(f"{args.export_dir}/model-*.safetensors"):
        os.remove(f)
    index_path = f"{args.export_dir}/model.safetensors.index.json"
    if os.path.exists(index_path):
        os.remove(index_path)

    tensors = {}
    for shard in sorted(glob.glob(f"{args.tmp_dir}/*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                tensors[f"thinker.{key}"] = f.get_tensor(key)

    save_file(tensors, f"{args.export_dir}/model.safetensors")
    print(f"[STEP 2] Done. vLLM checkpoint: {args.export_dir}")
    print(f"[INFO] Scale keys sample: {[k for k in tensors if 'scale' in k][:3]}")


if __name__ == "__main__":
    main()
