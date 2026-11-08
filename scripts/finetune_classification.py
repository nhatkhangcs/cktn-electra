#!/usr/bin/env python3
"""Fine-tune an encoder for 28-way category classification on the CKTN corpus.

Model selection:
    --model_key cktn                      released CKTN-ELECTRA (Hugging Face)
    --model_key cktn_init                 vocabulary-augmented RemBERT before pre-training
    --model_key mbert | xlmr | rembert    multilingual baselines
    --checkpoint path/discriminator_best.pt
                                          a local pre-training checkpoint (uses the config
                                          of --model_name_or_path or --model_key)

Examples:
    python scripts/finetune_classification.py --model_key cktn
    python scripts/finetune_classification.py --model_key xlmr --pooling mean
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from cktn_electra.data import read_records
from cktn_electra.downstream import classification as cls
from cktn_electra.downstream.encoders import BENCHMARK_MODELS, load_encoder, load_tokenizer
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, serializable, set_seed, setup_logging

logger = logging.getLogger("finetune_classification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_key", choices=sorted(BENCHMARK_MODELS), default="cktn")
    parser.add_argument("--model_name_or_path", default=None, help="Override the Hub id / local path for --model_key.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local CKTN-ELECTRA discriminator .pt")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--text_fields", nargs="+", default=["title", "content"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--class_weighting", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_examples", type=int, default=None, help="Debug: subsample train/dev.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.model_key if args.checkpoint is None else f"{args.model_key}_local"
    output_dir = args.output_dir or REPO_ROOT / "runs/downstream/classification" / args.pooling / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_records(args.corpus_dir, "train")
    dev_records = read_records(args.corpus_dir, "dev")
    if args.max_examples:
        train_records, dev_records = train_records[: args.max_examples], dev_records[: args.max_examples]
    label_to_id = cls.build_label_map(train_records)
    train_set = cls.CategoryDataset(train_records, label_to_id, args.text_fields)
    dev_set = cls.CategoryDataset(dev_records, label_to_id, args.text_fields)
    save_json(output_dir / "label_diagnostics.json", cls.label_diagnostics(label_to_id, train_records, dev_records))
    logger.info("train=%d dev=%d labels=%d", len(train_set), len(dev_set), len(label_to_id))

    tokenizer = load_tokenizer(args.model_key, args.model_name_or_path)
    collator = cls.CategoryCollator(tokenizer, args.max_length)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator, pin_memory=pin)
    dev_loader = DataLoader(dev_set, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator, pin_memory=pin)

    encoder, encoder_desc = load_encoder(args.model_key, args.model_name_or_path, args.checkpoint)
    logger.info("Encoder: %s", encoder_desc)
    weights = cls.balanced_class_weights(train_set, len(label_to_id)) if args.class_weighting == "balanced" else None
    model = cls.CategoryClassifier(encoder, len(label_to_id), args.dropout, args.pooling, weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best, report = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss = cls.train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_accum_steps, args.max_grad_norm)
        metrics = cls.evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "model_key": args.model_key,
            "encoder": encoder_desc,
            "pooling": args.pooling,
            "train_loss": round(train_loss, 6),
            "dev_loss": round(metrics["loss"], 6),
            **{k: round(metrics[k], 6) for k in ("accuracy", "macro_f1", "micro_f1", "weighted_f1")},
        }
        report.append(row)
        logger.info("epoch %d | loss %.4f | dev acc %.4f macro-F1 %.4f", epoch, train_loss, row["accuracy"], row["macro_f1"])
        if row["macro_f1"] > best:
            best = row["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_to_id": label_to_id,
                    "args": serializable(vars(args)),
                    "metrics": row,
                },
                output_dir / "best.pt",
            )
        save_json(output_dir / "report.json", report)
    save_json(output_dir / "label_to_id.json", label_to_id)
    logger.info("Best dev macro-F1 %.4f -> %s", best, output_dir)


if __name__ == "__main__":
    main()
