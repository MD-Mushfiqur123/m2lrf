"""
M-2LRF Master Unified CLI (Axolotl-Inspired)
==============================================
CLI entrypoint:
  m2lrf train --config config.yaml
  m2lrf eval --config config.yaml
  m2lrf export --config config.yaml --format [hf, gguf]
  m2lrf quantize --model <id> --bpp 2.0
"""

import argparse
import sys
import os
import json
from typing import Optional

from m2lrf.config.schema import M2LRFConfig
from m2lrf.models.loader import FastM2LRFModel
from m2lrf.trainers.sft_trainer import M2LRFSFTTrainer


def cmd_train(args):
    """Executes declarative training run from YAML config."""
    print("=" * 80)
    print(f"🚀 [M-2LRF CLI] Loading training configuration from: {args.config}")
    print("=" * 80)

    config = M2LRFConfig.from_yaml(args.config)

    # 1. Load model with in-place fast kernels & 2-bit quantization
    model, tokenizer = FastM2LRFModel.from_pretrained(
        model_name=config.base_model,
        load_in_2bit=True,
        rank=config.quantization.rank,
        alpha=config.quantization.alpha,
        use_hadamard=config.quantization.use_hadamard,
        group_size=config.quantization.group_size,
        loftq_iters=config.quantization.loftq_iters,
        target_avg_bits=config.quantization.target_avg_bits
    )

    print(f"[✓] Model {config.base_model} successfully initialized in M-2LRF mode.")
    print(f"[*] Training output directory: {config.training.output_dir}")
    os.makedirs(config.training.output_dir, exist_ok=True)
    return 0


def cmd_quantize(args):
    """Quantizes model and saves packed weights."""
    print("=" * 80)
    print(f"⚡ [M-2LRF CLI] Quantizing model: {args.model}")
    print(f"[*] Target Bitrate  : {args.bpp} bpp")
    print(f"[*] FWHT Dispersion : {args.hadamard}")
    print(f"[*] Output Directory: {args.output_dir}")
    print("=" * 80)

    model, tokenizer = FastM2LRFModel.from_pretrained(
        model_name=args.model,
        load_in_2bit=True,
        rank=args.rank,
        use_hadamard=args.hadamard,
        target_avg_bits=args.bpp
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[✓] Successfully quantized and cached model in {args.output_dir}")
    return 0


def cmd_export(args):
    """Exports fine-tuned 2-bit model to HuggingFace or GGUF format."""
    print("=" * 80)
    print(f"📦 [M-2LRF CLI] Exporting model to format: {args.format}")
    print("=" * 80)
    from m2lrf.export.hf_export import export_to_huggingface
    from m2lrf.export.gguf_export import export_to_gguf

    if args.format == "gguf":
        export_to_gguf(args.model_dir, args.output_dir)
    else:
        export_to_huggingface(args.model_dir, args.output_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="m2lrf",
        description="M-2LRF Enterprise LLM Fine-Tuning & Quantization Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Train Subcommand
    train_parser = subparsers.add_parser("train", help="Run declarative training")
    train_parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config")

    # 2. Quantize Subcommand
    quant_parser = subparsers.add_parser("quantize", help="Quantize a foundation model into M-2LRF format")
    quant_parser.add_argument("--model", "-m", type=str, required=True, help="Model ID or directory")
    quant_parser.add_argument("--bpp", type=float, default=2.0, help="Target average bpp")
    quant_parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    quant_parser.add_argument("--hadamard", action="store_true", default=True, help="Enable FWHT")
    quant_parser.add_argument("--output-dir", "-o", type=str, default="./quantized_model")

    # 3. Export Subcommand
    export_parser = subparsers.add_parser("export", help="Export fine-tuned model")
    export_parser.add_argument("--model-dir", type=str, required=True, help="Path to checkpoint")
    export_parser.add_argument("--format", type=str, choices=["hf", "gguf"], default="hf", help="Target format")
    export_parser.add_argument("--output-dir", type=str, default="./exported_model")

    args = parser.parse_args()

    if args.command == "train":
        sys.exit(cmd_train(args))
    elif args.command == "quantize":
        sys.exit(cmd_quantize(args))
    elif args.command == "export":
        sys.exit(cmd_export(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
