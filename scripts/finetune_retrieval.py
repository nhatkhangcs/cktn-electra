#!/usr/bin/env python3
"""Fine-tune a bi-encoder for summary-to-document retrieval on the CKTN corpus.

Queries are article summaries (falling back to tags); documents are title + content.
Evaluation ranks every development query against the full multilingual
development document pool (exact dot-product search).

Examples:
    python scripts/finetune_retrieval.py --model_key cktn
    python scripts/finetune_retrieval.py --model_key rembert
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from cktn_electra.data import read_records
from cktn_electra.downstream import retrieval as ir
from cktn_electra.downstream.encoders import BENCHMARK_MODELS, load_encoder, load_tokenizer
from cktn_electra.modeling import decay_parameter_groups
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, serializable, set_seed, setup_logging

logger = logging.getLogger("finetune_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_key", choices=sorted(BENCHMARK_MODELS), default="cktn")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local CKTN-ELECTRA discriminator .pt")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--query_fields", nargs="+", default=["summary"])
    parser.add_argument("--fallback_query_fields", nargs="*", default=["tags"])
    parser.add_argument("--doc_fields", nargs="+", default=["title", "content"])
    parser.add_argument("--max_query_length", type=int, default=512)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_group_key", default="_source", help="Record field used to group batches ('none' to disable).")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_examples", type=int, default=None, help="Debug: subsample train/dev.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.batch_size < 2:
        raise SystemExit("In-batch contrastive training needs --batch_size >= 2.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.model_key if args.checkpoint is None else f"{args.model_key}_local"
    output_dir = args.output_dir or REPO_ROOT / "runs/downstream/retrieval" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_records(args.corpus_dir, "train")
    dev_records = read_records(args.corpus_dir, "dev")
    if args.max_examples:
        train_records, dev_records = train_records[: args.max_examples], dev_records[: args.max_examples]
    train_set = ir.RetrievalPairDataset(train_records, args.query_fields, args.fallback_query_fields, args.doc_fields, args.batch_group_key)
    eval_sets = ir.build_eval_sets(dev_records, args.query_fields, args.fallback_query_fields, args.doc_fields)
    logger.info(
        "train pairs=%d | dev queries=%d documents=%d | groups=%s",
        len(train_set), len(eval_sets["queries"]), len(eval_sets["documents"]), dict(train_set.groups),
    )

    tokenizer = load_tokenizer(args.model_key, args.model_name_or_path)
    train_loader = DataLoader(
        train_set,
        batch_sampler=ir.GroupedBatchSampler(train_set.examples, args.batch_size, drop_last=True),
        num_workers=args.num_workers,
        collate_fn=ir.RetrievalCollator(tokenizer, args.max_query_length, args.max_doc_length),
        pin_memory=device.type == "cuda",
    )

    encoder, encoder_desc = load_encoder(args.model_key, args.model_name_or_path, args.checkpoint)
    logger.info("Encoder: %s", encoder_desc)
    model = ir.BiEncoderRetriever(encoder, args.temperature).to(device)
    optimizer = torch.optim.AdamW(decay_parameter_groups(model, args.weight_decay), lr=args.lr)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best, report = -1.0, []
    for epoch in range(1, args.epochs + 1):
        train_loss = ir.train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_accum_steps, args.max_grad_norm)
        metrics = ir.evaluate(model, tokenizer, eval_sets, args.max_query_length, args.max_doc_length, args.eval_batch_size, device)
        row = {
            "epoch": epoch,
            "model_key": args.model_key,
            "encoder": encoder_desc,
            "train_loss": round(train_loss, 6),
            **{k: round(metrics[k], 6) for k in ("mrr_at_10", "recall_at_1", "recall_at_5", "recall_at_10", "ndcg_at_10")},
            "num_queries": metrics["num_queries"],
            "num_documents": metrics["num_documents"],
        }
        report.append(row)
        logger.info("epoch %d | loss %.4f | MRR@10 %.4f R@10 %.4f", epoch, train_loss, row["mrr_at_10"], row["recall_at_10"])
        if row["mrr_at_10"] > best:
            best = row["mrr_at_10"]
            torch.save({"model": model.state_dict(), "args": serializable(vars(args)), "metrics": row}, output_dir / "best.pt")
        save_json(output_dir / "report.json", report)
    logger.info("Best dev MRR@10 %.4f -> %s", best, output_dir)


if __name__ == "__main__":
    main()
