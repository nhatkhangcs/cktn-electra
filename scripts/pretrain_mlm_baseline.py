#!/usr/bin/env python3
"""MLM continued pre-training (or MLM evaluation only) of a baseline encoder on the CKTN corpus.

Examples:
    python scripts/pretrain_mlm_baseline.py --model_key xlmr --epochs 5 --lr 2e-5
    python scripts/pretrain_mlm_baseline.py --model_key rembert --eval_only
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from cktn_electra import mlm_baseline as mlm
from cktn_electra.modeling import decay_parameter_groups
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, serializable, set_seed, setup_logging

logger = logging.getLogger("pretrain_mlm_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_key", choices=sorted(mlm.BASELINE_MODELS), required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "runs/cache/mlm")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--fields", nargs="+", default=["title", "summary", "tags", "content"])
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--min_tokens", type=int, default=8)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_only", action="store_true", help="Report MLM loss/perplexity without training.")
    parser.add_argument("--save_each_epoch", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_or_build(args, tokenizer, split: str, model_id: str):
    name = f"{args.model_key}_{model_id.replace('/', '__')}_{split}_seq{args.max_seq_length}_min{args.min_tokens}_{'-'.join(args.fields)}.pt"
    path = args.cache_dir / name
    if path.exists() and not args.rebuild_cache:
        return torch.load(path, map_location="cpu", weights_only=True)["examples"]
    examples = mlm.build_token_chunks(
        tokenizer, mlm.iter_split_texts(args.corpus_dir, split, args.fields), args.max_seq_length, args.min_tokens
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": examples, "model_id": model_id, "split": split}, path)
    return examples


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad_accum_steps must be >= 1")
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.fp16 and device.type == "cuda"
    output_dir = args.output_dir or REPO_ROOT / "runs/mlm_baselines" / args.model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_name_or_path or mlm.BASELINE_MODELS[args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=args.model_key != "rembert")
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    dev_loader = DataLoader(
        mlm.MLMDataset(load_or_build(args, tokenizer, "dev", model_id)),
        batch_size=args.eval_batch_size or args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    if args.eval_only:
        metrics = mlm.evaluate(model, dev_loader, device, use_amp)
        save_json(output_dir / "eval_only_report.json", {"model_id": model_id, **metrics})
        logger.info("%s | dev MLM loss %.4f | PPL %.2f", model_id, metrics["loss"], metrics["perplexity"])
        return

    train_loader = DataLoader(
        mlm.MLMDataset(load_or_build(args, tokenizer, "train", model_id)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(decay_parameter_groups(model, args.weight_decay), lr=args.lr)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    report, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = mlm.train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp, args.grad_accum_steps, args.max_grad_norm)
        metrics = mlm.evaluate(model, dev_loader, device, use_amp)
        row = {"epoch": epoch, "model_id": model_id, "train_loss": round(train_loss, 6), "eval_loss": round(metrics["loss"], 6), "perplexity": metrics["perplexity"]}
        report.append(row)
        save_json(output_dir / "report.json", report)
        logger.info("epoch %d | train %.4f | dev %.4f | PPL %.2f", epoch, train_loss, metrics["loss"], metrics["perplexity"])
        if metrics["loss"] < best:
            best = metrics["loss"]
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
        if args.save_each_epoch:
            model.save_pretrained(output_dir / f"checkpoint-epoch{epoch}")
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    save_json(output_dir / "training_args.json", serializable(vars(args)))


if __name__ == "__main__":
    main()
