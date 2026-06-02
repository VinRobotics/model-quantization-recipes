# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import torch
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",         required=True)
    parser.add_argument("--output_path",        required=True)
    parser.add_argument("--num_calib_samples",  type=int, default=512)
    parser.add_argument("--max_seq_len",        type=int, default=1024)
    parser.add_argument("--dataset_id",         default="abisee/cnn_dailymail")
    parser.add_argument("--dataset_config",     default="3.0.0")
    parser.add_argument("--dataset_split",      default="train")
    parser.add_argument("--device",             default="auto")
    parser.add_argument("--max_memory_per_gpu", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{args.max_memory_per_gpu}GiB" for i in range(num_gpus)}
    print(f"GPUs: {num_gpus}, VRAM cap: {args.max_memory_per_gpu} GiB each")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device,
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Quantize LLM backbone only (196 Linear layers, 28 layers x 7)
    # Visual encoder (blocks 0-23 + merger + deepstack) kept in bf16
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*visual.*",
            "re:.*lm_head",
        ],
    )

    # Text-only calib: vision bị ignore hoàn toàn -> multimodal không cần thiết
    print(f"Loading dataset '{args.dataset_id}' (streaming)...")
    ds = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=args.num_calib_samples * 3).take(args.num_calib_samples)

    samples = []
    print(f"Tokenizing {args.num_calib_samples} samples: ", end="", flush=True)
    for i, item in enumerate(ds):
        if (i + 1) % 50 == 0:
            print(f"{i + 1} ", end="", flush=True)
        text = item["article"].strip()
        inputs = processor.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        samples.append({
            "input_ids":      inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
        })
    print(f"\nLoaded {len(samples)} samples.")

    calib_dataset = Dataset.from_list(samples)

    def data_collator(batch):
        assert len(batch) == 1
        return {key: torch.tensor(value).unsqueeze(0) for key, value in batch[0].items()}

    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_dataset,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calib_samples,
        data_collator=data_collator,
    )

    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print(f"Done. Saved to {args.output_path}")


if __name__ == "__main__":
    main()
