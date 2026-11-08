"""Masked-language-model continued pre-training for the baseline encoders (mBERT, XLM-R, RemBERT)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from tqdm import tqdm

from cktn_electra.data import field_to_text

logger = logging.getLogger(__name__)

BASELINE_MODELS = {
    "rembert": "google/rembert",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
}


def iter_split_texts(corpus_dir: Path, split: str, fields: Sequence[str]) -> Iterable[str]:
    """Yield one text per record: the given fields joined by spaces."""
    import json

    split_dir = Path(corpus_dir) / split
    paths = sorted(split_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files under {split_dir}")
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                text = " ".join(p for p in (field_to_text(record.get(f, "")) for f in fields) if p)
                if text:
                    yield text


def _special_ids(tokenizer):
    prefix = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    suffix = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    return ([prefix] if prefix is not None else []), ([suffix] if suffix is not None else [])


def build_token_chunks(tokenizer, texts: Iterable[str], max_seq_length: int, min_tokens: int) -> List[List[int]]:
    """Non-overlapping chunks with model-specific boundary tokens; chunks shorter than ``min_tokens`` are dropped."""
    prefix, suffix = _special_ids(tokenizer)
    block = max_seq_length - len(prefix) - len(suffix)
    if block <= 0:
        raise ValueError(f"max_seq_length={max_seq_length} is too small for this tokenizer")
    examples = []
    for text in tqdm(texts, desc="Tokenizing", unit="doc"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for start in range(0, len(ids), block):
            chunk = ids[start : start + block]
            if len(chunk) >= min_tokens:
                examples.append(prefix + chunk + suffix)
    return examples


class MLMDataset(Dataset):
    def __init__(self, examples: Sequence[Sequence[int]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Sequence[int]]:
        return {"input_ids": self.examples[index]}


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp: bool, grad_accum_steps: int, max_grad_norm: float) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = model(**batch).loss
        scaler.scale(loss / grad_accum_steps).backward()
        total += loss.item()
        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return total / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    total = 0.0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            total += model(**batch).loss.item()
    loss = total / max(len(dataloader), 1)
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 20 else float("inf")}
