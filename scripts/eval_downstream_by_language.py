#!/usr/bin/env python3
"""Per-language breakdown of fine-tuned classification and retrieval checkpoints.

Retrieval keeps the full multilingual candidate pool: each query is ranked once
against all development documents, and ranks are then grouped by query language.

Examples:
    python scripts/eval_downstream_by_language.py --task classification \
        --checkpoint runs/downstream/classification/cls/cktn/best.pt
    python scripts/eval_downstream_by_language.py --task retrieval \
        --checkpoint runs/downstream/retrieval/cktn/best.pt
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

from torch.utils.data import DataLoader

from cktn_electra.data import LANGUAGE_LABELS, LANGUAGES, read_records
from cktn_electra.downstream import classification as cls
from cktn_electra.downstream import retrieval as ir
from cktn_electra.downstream.encoders import empty_encoder, load_tokenizer
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, set_seed, setup_logging, torch_load


def encoder_settings(saved_args: Dict):
    """Model key / path recorded by the fine-tuning script (legacy checkpoints default to ``cktn``)."""
    path = saved_args.get("model_name_or_path")
    path = None if path in (None, "", "None") else str(path)
    return str(saved_args.get("model_key", "cktn")), path


def write_rows(stem: Path, metadata: Dict, rows: List[Dict]) -> None:
    save_json(stem.with_suffix(".json"), {"metadata": metadata, "results": rows})
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_classification(checkpoint: Dict, records: List[Dict], args, device) -> List[Dict]:
    saved = dict(checkpoint.get("args", {}))
    model_key, model_path = encoder_settings(saved)
    label_to_id = dict(checkpoint["label_to_id"])
    state = checkpoint["model"]
    model = cls.CategoryClassifier(
        empty_encoder(model_key, model_path),
        len(label_to_id),
        float(saved.get("dropout", 0.1)),
        str(saved.get("pooling", "cls")),
        state.get("loss_fn.weight"),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    collator = cls.CategoryCollator(load_tokenizer(model_key, model_path), int(saved.get("max_length", 512)))
    fields = list(saved.get("text_fields", ["title", "content"]))

    rows, all_labels, all_preds = [], [], []
    for language in LANGUAGES:
        dataset = cls.CategoryDataset([r for r in records if r["_language"] == language], label_to_id, fields)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers)
        out = cls.predict(model, loader, device)
        rows.append({"language": LANGUAGE_LABELS[language], "num_examples": len(out["labels"]),
                     **cls.classification_metrics(out["labels"], out["preds"])})
        all_labels += out["labels"]
        all_preds += out["preds"]
    rows.append({"language": "Overall", "num_examples": len(all_labels), **cls.classification_metrics(all_labels, all_preds)})
    return rows


def evaluate_retrieval(checkpoint: Dict, records: List[Dict], args, device) -> List[Dict]:
    saved = dict(checkpoint.get("args", {}))
    model_key, model_path = encoder_settings(saved)
    model = ir.BiEncoderRetriever(empty_encoder(model_key, model_path), float(saved.get("temperature", 0.1)))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    tokenizer = load_tokenizer(model_key, model_path)

    eval_sets = ir.build_eval_sets(
        records,
        list(saved.get("query_fields", ["summary"])),
        list(saved.get("fallback_query_fields", ["tags"])),
        list(saved.get("doc_fields", ["title", "content"])),
    )
    metrics = ir.evaluate(
        model, tokenizer, eval_sets,
        int(saved.get("max_query_length", 512)), int(saved.get("max_doc_length", 512)), args.batch_size, device,
    )
    ranks = metrics["ranks"]
    query_langs = [q["language"] for q in eval_sets["queries"]]
    doc_counts = Counter(d["language"] for d in eval_sets["documents"])
    rows = []
    for language in LANGUAGES:
        idx = [i for i, lang in enumerate(query_langs) if lang == language]
        rows.append({"language": LANGUAGE_LABELS[language], "num_queries": len(idx),
                     "language_documents": doc_counts[language], **ir.ranks_to_metrics(ranks[idx])})
    rows.append({"language": "Overall", "num_queries": len(ranks),
                 "language_documents": len(eval_sets["documents"]), **ir.ranks_to_metrics(ranks)})
    return rows


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["classification", "retrieval"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by a fine-tuning script")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--output_dir", type=Path, default=None, help="Default: next to the checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(42)
    device = resolve_device(args.device)
    checkpoint = torch_load(args.checkpoint)
    records = read_records(args.corpus_dir, "dev")
    evaluate = evaluate_classification if args.task == "classification" else evaluate_retrieval
    rows = evaluate(checkpoint, records, args, device)

    stem = (args.output_dir or args.checkpoint.parent) / f"{args.task}_per_language"
    write_rows(stem, {"task": args.task, "checkpoint": str(args.checkpoint), "checkpoint_metrics": checkpoint.get("metrics", {})}, rows)
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
