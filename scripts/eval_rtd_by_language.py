#!/usr/bin/env python3
"""Replaced-token-detection diagnostics per language for one or more pre-training runs.

Each language partition of the development split is evaluated separately; the
overall row is obtained by merging token-level counts.

Example:
    python scripts/eval_rtd_by_language.py \
        --run main=runs/pretrain/main --run no_script_aware=runs/pretrain/no_script_aware \
        --checkpoint latest
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cktn_electra.checkpoints import RunPaths, load_model_weights, resolve_tag
from cktn_electra.config import PretrainConfig, load_pretrain_config
from cktn_electra.data import (
    LANGUAGE_LABELS,
    LANGUAGES,
    ELECTRADataset,
    load_or_build_chunks,
    load_pretrain_tokenizer,
    split_files,
)
from cktn_electra.modeling import CKTNElectra
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, seed_worker, set_seed, setup_logging

METRICS = (
    "avg_mlm_loss", "avg_rtd_loss", "accuracy", "f1", "replacement_rate",
    "effective_token_replacement_rate", "valid_candidate_rate", "disc_confidence", "rtd_entropy",
)


@dataclass
class Counts:
    mlm_loss_sum: float = 0.0
    rtd_loss_sum: float = 0.0
    replaced: float = 0.0
    valid_candidates: float = 0.0
    confidence_sum: float = 0.0
    entropy_sum: float = 0.0
    masked: int = 0
    tokens: int = 0
    correct: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def metrics(self) -> Dict[str, float]:
        masked, tokens = max(self.masked, 1), max(self.tokens, 1)
        denom = 2 * self.tp + self.fp + self.fn
        return {
            "avg_mlm_loss": self.mlm_loss_sum / masked,
            "avg_rtd_loss": self.rtd_loss_sum / tokens,
            "accuracy": self.correct / tokens,
            "f1": 2 * self.tp / denom if denom else 0.0,
            "replacement_rate": self.replaced / masked,  # replaced / MLM-selected tokens
            "effective_token_replacement_rate": self.replaced / tokens,  # replaced / non-padding tokens
            "valid_candidate_rate": self.valid_candidates / masked,
            "disc_confidence": self.confidence_sum / tokens,
            "rtd_entropy": self.entropy_sum / tokens,
        }


@torch.inference_mode()
def evaluate_once(model, dataset, seed: int, batch_size: int, num_workers: int, device, lam: float, desc: str) -> Counts:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
    totals = Counts()
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, lam=lam)
        masked = int(((batch["labels"] != -100) & batch["attention_mask"].bool()).sum())
        mask = out["rtd_label_mask"]
        tokens = int(mask.sum())
        if masked == 0 or tokens == 0:
            continue
        pred = (torch.sigmoid(out["disc_logits"].squeeze(-1)) > 0.5).long()[mask]
        gold = out["rtd_labels"].long()[mask]
        true_rep, pred_rep = gold == 0, pred == 0
        totals.mlm_loss_sum += out["loss_mlm"].item() * masked
        totals.rtd_loss_sum += out["loss_disc"].item() * tokens
        totals.replaced += out["replacement_mask"].sum().item()
        totals.valid_candidates += out["valid_candidate_rate"].item() * masked
        totals.confidence_sum += out["avg_disc_confidence"].item() * tokens
        totals.entropy_sum += out["rtd_entropy"].item() * tokens
        totals.masked += masked
        totals.tokens += tokens
        totals.correct += int((pred == gold).sum())
        totals.tp += int((pred_rep & true_rep).sum())
        totals.fp += int((pred_rep & ~true_rep).sum())
        totals.fn += int((~pred_rep & true_rep).sum())
    return totals


def summarize(runs: List[Counts], language: str, chunks: int) -> Dict:
    per_run = [r.metrics() for r in runs]
    row = {"language": language, "language_label": LANGUAGE_LABELS[language], "chunks": chunks}
    for metric in METRICS:
        values = [r[metric] for r in per_run]
        row[metric] = fmean(values)
        row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return row


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR",
                        help="Run to evaluate; the config is read from RUN_DIR/checkpoint/config.json "
                             "or configs/pretrain/NAME.json.")
    parser.add_argument("--checkpoint", default="latest", help="best, latest or an epoch number")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "runs/cache/chunks")
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat stochastic masking/sampling; report mean/std.")
    parser.add_argument("--max_chunks", type=int, default=0, help="Debug: chunks per language (0 = all).")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "runs/rtd_by_language")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)

    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        run_dir = Path(run_dir)
        saved = run_dir / "checkpoint" / "config.json"
        config: PretrainConfig = load_pretrain_config(saved if saved.exists() else REPO_ROOT / f"configs/pretrain/{name}.json")
        tokenizer = load_pretrain_tokenizer(config.base_model)
        datasets = {}
        for language in args.languages:
            files = [p for p in split_files(args.corpus_dir, "dev") if p.name.startswith(language + "_")]
            chunks = load_or_build_chunks(f"dev_{language}", files, tokenizer, config.seq_len, args.cache_dir, config.base_model)
            if args.max_chunks:
                chunks = chunks[: args.max_chunks]
            datasets[language] = ELECTRADataset(chunks, tokenizer, config.seq_len, config.mask_rate)

        paths = RunPaths(run_dir)
        tag = resolve_tag(paths, args.checkpoint)
        model = CKTNElectra(config, tokenizer=tokenizer, load_pretrained=False)
        load_model_weights(model, paths.disc_path(tag), paths.gen_path(tag))
        model.to(device).eval()

        per_language = {lang: [] for lang in args.languages}
        overall_runs = []
        for repeat in range(args.repeats):
            overall = Counts()
            for index, language in enumerate(args.languages):
                counts = evaluate_once(
                    model, datasets[language], args.seed + repeat * 10_000 + index,
                    args.batch_size, args.num_workers, device, config.lambda_max, f"{name}/{language}#{repeat + 1}",
                )
                per_language[language].append(counts)
                overall.add(counts)
            overall_runs.append(overall)

        rows = [summarize(per_language[lang], lang, len(datasets[lang])) for lang in args.languages]
        rows.append(summarize(overall_runs, "overall", sum(len(d) for d in datasets.values())))
        stem = args.output_dir / f"{name}_{tag}"
        save_json(stem.with_suffix(".json"), {"run": name, "checkpoint": tag, "config": config.to_dict(), "results": rows})
        with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            print(f"{name:<18} {row['language_label']:<9} acc={row['accuracy']:.4f} f1={row['f1']:.4f} "
                  f"rep={row['replacement_rate']:.4f} valid={row['valid_candidate_rate']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
