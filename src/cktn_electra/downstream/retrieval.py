"""Summary-to-document retrieval with a bi-encoder trained by in-batch contrastive loss."""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from cktn_electra.data import join_fields
from cktn_electra.downstream.encoders import accepts_token_type_ids, masked_mean_pool, move_to

logger = logging.getLogger(__name__)


def select_query(record: Dict, query_fields: Sequence[str], fallback_fields: Sequence[str]) -> Tuple[str, str]:
    """Return ``(query, source)`` where source is ``primary``, ``fallback`` or ``missing``."""
    query = join_fields(record, query_fields)
    if query:
        return query, "primary"
    fallback = join_fields(record, fallback_fields)
    if fallback:
        return fallback, "fallback"
    return "", "missing"


class RetrievalPairDataset(Dataset):
    """(query, positive document) pairs, each tagged with a batching group."""

    def __init__(
        self,
        records: Sequence[Dict],
        query_fields: Sequence[str],
        fallback_fields: Sequence[str],
        doc_fields: Sequence[str],
        group_key: str = "_source",
    ):
        self.examples: List[Dict] = []
        self.query_sources: Counter = Counter()
        self.groups: Counter = Counter()
        for record in records:
            query, source = select_query(record, query_fields, fallback_fields)
            document = join_fields(record, doc_fields)
            self.query_sources[source] += 1
            if not (query and document):
                continue
            group = "all" if group_key == "none" else (str(record.get(group_key, "")).strip() or "all")
            self.groups[group] += 1
            self.examples.append({"id": record["_id"], "query": query, "document": document, "group": group})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        return self.examples[index]


class GroupedBatchSampler(Sampler[List[int]]):
    """Batches drawn from a single group (by default: one source file / language).

    Keeping each batch within one language stops the contrastive loss from
    being solved by language identity alone.
    """

    def __init__(self, examples: Sequence[Dict], batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_to_indices: Dict[str, List[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            self.group_to_indices[example["group"]].append(index)

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for indices in self.group_to_indices.values():
            shuffled = indices[:]
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for indices in self.group_to_indices.values():
            total += len(indices) // self.batch_size
            total += int(not self.drop_last and len(indices) % self.batch_size > 0)
        return total


class RetrievalCollator:
    def __init__(self, tokenizer, max_query_length: int, max_doc_length: int):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def _encode(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        return dict(self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"))

    def __call__(self, batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "query": self._encode([item["query"] for item in batch], self.max_query_length),
            "document": self._encode([item["document"] for item in batch], self.max_doc_length),
        }


class BiEncoderRetriever(nn.Module):
    """Shared encoder, masked-mean pooling, L2 normalisation, symmetric InfoNCE loss."""

    def __init__(self, encoder: nn.Module, temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.accepts_token_type_ids = accepts_token_type_ids(encoder)

    def encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        pooled = masked_mean_pool(self.encoder(**kwargs).last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]):
        query_emb = self.encode(**query)
        doc_emb = self.encode(**document)
        logits = query_emb @ doc_emb.t() / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
        return {"loss": loss, "logits": logits, "query_emb": query_emb, "doc_emb": doc_emb}


def build_eval_sets(
    records: Sequence[Dict],
    query_fields: Sequence[str],
    fallback_fields: Sequence[str],
    doc_fields: Sequence[str],
) -> Dict:
    """Every record with a document is a candidate; every record with a query is a query."""
    documents, doc_index = [], {}
    for record in records:
        text = join_fields(record, doc_fields)
        if text:
            doc_index[record["_id"]] = len(documents)
            documents.append({"id": record["_id"], "text": text, "language": record.get("_language")})

    queries, sources = [], Counter()
    for record in records:
        query, source = select_query(record, query_fields, fallback_fields)
        sources[source] += 1
        if query and record["_id"] in doc_index:
            queries.append(
                {
                    "id": record["_id"],
                    "text": query,
                    "positive_index": doc_index[record["_id"]],
                    "language": record.get("_language"),
                }
            )
    return {"queries": queries, "documents": documents, "query_sources": dict(sources)}


@torch.no_grad()
def encode_texts(model: BiEncoderRetriever, tokenizer, texts: Sequence[str], max_length: int, batch_size: int, device) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        chunks.append(model.encode(**move_to(dict(encoded), device)).cpu())
    model.train()
    return torch.cat(chunks, dim=0)


def compute_ranks(query_emb: torch.Tensor, doc_emb: torch.Tensor, positives: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """1-based rank of each query's positive document among all candidates (exact search)."""
    ranks = []
    for start in range(0, len(query_emb), batch_size):
        scores = query_emb[start : start + batch_size] @ doc_emb.t()
        positive_scores = scores.gather(1, positives[start : start + batch_size].unsqueeze(1))
        ranks.append((scores > positive_scores).sum(dim=1) + 1)
    return torch.cat(ranks)


def ranks_to_metrics(ranks: torch.Tensor) -> Dict[str, float]:
    ranks = ranks.float()
    top10 = ranks <= 10
    return {
        "mrr_at_10": float(torch.where(top10, 1.0 / ranks, torch.zeros_like(ranks)).mean()),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "recall_at_10": float(top10.float().mean()),
        "ndcg_at_10": float(torch.where(top10, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks)).mean()),
    }


def evaluate(model, tokenizer, eval_sets: Dict, max_query_length: int, max_doc_length: int, batch_size: int, device) -> Dict:
    queries, documents = eval_sets["queries"], eval_sets["documents"]
    query_emb = encode_texts(model, tokenizer, [q["text"] for q in queries], max_query_length, batch_size, device)
    doc_emb = encode_texts(model, tokenizer, [d["text"] for d in documents], max_doc_length, batch_size, device)
    positives = torch.tensor([q["positive_index"] for q in queries], dtype=torch.long)
    ranks = compute_ranks(query_emb, doc_emb, positives)
    return {
        **ranks_to_metrics(ranks),
        "num_queries": len(queries),
        "num_documents": len(documents),
        "ranks": ranks,
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps: int, max_grad_norm: float) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training"), start=1):
        loss = model(**move_to(batch, device))["loss"]
        (loss / grad_accum_steps).backward()
        total_loss += loss.item()
        if step % grad_accum_steps == 0 or step == len(dataloader):
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return total_loss / max(len(dataloader), 1)
