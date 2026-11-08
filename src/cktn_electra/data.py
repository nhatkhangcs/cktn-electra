"""CKTN corpus loading, tokenisation/chunking with an on-disk cache, and MLM masking.

Expected corpus layout::

    corpus/
      train/  cham_train.json  khmer_train.json  tay_nung_train.json
      dev/    cham_dev.json    khmer_dev.json    tay_nung_dev.json

Each file is a JSON list of records with the fields
``site, category, url, title, summary, tags, content``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

LANGUAGES: Tuple[str, ...] = ("cham", "khmer", "tay_nung")
LANGUAGE_LABELS = {"cham": "Cham", "khmer": "Khmer", "tay_nung": "Tay-Nung", "overall": "Overall"}
SPLITS = ("train", "dev")


# --------------------------------------------------------------------------- #
# Raw records
# --------------------------------------------------------------------------- #

def split_files(corpus_dir: Path, split: str, languages: Sequence[str] = LANGUAGES) -> List[Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    return [Path(corpus_dir) / split / f"{language}_{split}.json" for language in languages]


def read_records(corpus_dir: Path, split: str) -> List[Dict]:
    """Read all records of a split, tagging each with ``_source``, ``_id`` and ``_language``."""
    records: List[Dict] = []
    for language, path in zip(LANGUAGES, split_files(corpus_dir, split)):
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for index, item in enumerate(data):
            item = dict(item)
            item["_source"] = path.name
            item["_index"] = index
            item["_id"] = f"{path.stem}:{index}"
            item["_language"] = language
            records.append(item)
    return records


def field_to_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def join_fields(record: Dict, fields: Sequence[str], separator: str = "\n") -> str:
    pieces = [field_to_text(record.get(field, "")) for field in fields]
    return separator.join(piece for piece in pieces if piece)


def read_contents(json_files: Iterable[Path]) -> List[str]:
    """Return the non-empty ``content`` field of every record in ``json_files``."""
    contents: List[str] = []
    for path in json_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]
        contents.extend(text for text in (item.get("content", "").strip() for item in items) if text)
        logger.info("Loaded %s (%d documents so far)", path.name, len(contents))
    return contents


# --------------------------------------------------------------------------- #
# Tokenisation and caching
# --------------------------------------------------------------------------- #

def tokenize_and_chunk(
    contents: Iterable[str], tokenizer: PreTrainedTokenizerBase, max_len: int
) -> List[List[int]]:
    """Split each document into non-overlapping ``[CLS] ... [SEP]`` chunks of at most ``max_len``."""
    body_len = max_len - 2
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    chunks: List[List[int]] = []
    for text in tqdm(contents, desc="Tokenizing", leave=False):
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for start in range(0, len(token_ids), body_len):
            body = token_ids[start : start + body_len]
            if body:
                chunks.append([cls_id, *body, sep_id])
    logger.info("Built %d chunks", len(chunks))
    return chunks


def cache_fingerprint(
    json_files: Sequence[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    base_model: str,
) -> str:
    """Hash of everything that determines the chunked output.

    File paths are stored relative to the corpus directory's parent (e.g.
    ``corpus/train/cham_train.json``), together with size and mtime, so a
    modified corpus invalidates the cache.
    """
    payload = {
        "checkpoint": base_model,
        "tokenizer_name": getattr(tokenizer, "name_or_path", base_model),
        "vocab_size": tokenizer.vocab_size,
        "max_len": max_len,
        "files": [],
    }
    for path in json_files:
        path = Path(path).resolve()
        entry = {"path": str(path.relative_to(path.parents[2]))}
        if path.exists():
            stat = path.stat()
            entry["size"] = stat.st_size
            entry["mtime_ns"] = stat.st_mtime_ns
        else:
            entry["missing"] = True
        payload["files"].append(entry)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_or_build_chunks(
    name: str,
    json_files: Sequence[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    cache_dir: Path,
    base_model: str,
    rebuild: bool = False,
) -> List[List[int]]:
    """Return chunked token ids for ``json_files``, reusing ``cache_dir/<name>_<hash>.pkl``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = cache_fingerprint(json_files, tokenizer, max_len, base_model)
    cache_path = cache_dir / f"{name}_{fingerprint}.pkl"

    if cache_path.exists() and not rebuild:
        with open(cache_path, "rb") as handle:
            chunks = pickle.load(handle)["chunks"]
        logger.info("Loaded %d %s chunks from %s", len(chunks), name, cache_path)
        return chunks

    chunks = tokenize_and_chunk(read_contents(json_files), tokenizer, max_len)
    with open(cache_path, "wb") as handle:
        pickle.dump({"chunks": chunks}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d %s chunks to %s", len(chunks), name, cache_path)
    return chunks


# --------------------------------------------------------------------------- #
# Masked dataset
# --------------------------------------------------------------------------- #

class ELECTRADataset(Dataset):
    """Pads each chunk to ``max_len`` and masks ``mask_rate`` of its non-special tokens.

    Every selected position is replaced by ``[MASK]`` (no 80/10/10 split), as
    in ELECTRA's generator input. Masks are re-drawn every time an item is read.
    """

    def __init__(
        self,
        chunks: Sequence[Sequence[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        mask_rate: float,
    ):
        self.chunks = chunks
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def __len__(self) -> int:
        return len(self.chunks)

    def apply_mlm(self, token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        masked_ids = list(token_ids)
        labels = [-100] * len(token_ids)
        eligible = [i for i, tid in enumerate(token_ids) if tid not in self.special_ids]
        if not eligible:
            return masked_ids, labels

        n_mask = max(1, int(round(len(eligible) * self.mask_rate)))
        for idx in np.random.choice(eligible, size=n_mask, replace=False).tolist():
            labels[idx] = token_ids[idx]
            masked_ids[idx] = self.mask_id
        return masked_ids, labels

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        masked_ids, labels = self.apply_mlm(self.chunks[index])
        seq_len = len(masked_ids)
        pad_len = self.max_len - seq_len
        return {
            "input_ids": torch.tensor(masked_ids + [self.pad_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
            "token_type_ids": torch.zeros(self.max_len, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad_len, dtype=torch.long),
        }


def load_pretrain_tokenizer(base_model: str, expected_vocab_size: Optional[int] = None):
    """Load the slow (SentencePiece) tokenizer and check it matches the model vocabulary."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if expected_vocab_size is None:
        expected_vocab_size = AutoConfig.from_pretrained(base_model).vocab_size
    if tokenizer.vocab_size != expected_vocab_size:
        raise RuntimeError(
            f"Vocab size mismatch for {base_model}: tokenizer has {tokenizer.vocab_size} "
            f"entries but the model expects {expected_vocab_size}."
        )
    return tokenizer
