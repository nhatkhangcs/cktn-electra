"""Category classification: fine-tune an encoder with a linear head over title + content."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm

from cktn_electra.data import join_fields
from cktn_electra.downstream.encoders import accepts_token_type_ids, masked_mean_pool, move_to

logger = logging.getLogger(__name__)


def build_label_map(records: Sequence[Dict]) -> Dict[str, int]:
    labels = sorted({r.get("category", "").strip() for r in records if r.get("category", "").strip()})
    return {label: index for index, label in enumerate(labels)}


def category_counter(records: Sequence[Dict]) -> Counter:
    return Counter(r.get("category", "").strip() for r in records if r.get("category", "").strip())


class CategoryDataset(Dataset):
    """Records whose category is in ``label_to_id`` and whose text fields are non-empty."""

    def __init__(self, records: Sequence[Dict], label_to_id: Dict[str, int], text_fields: Sequence[str]):
        self.examples = []
        for record in records:
            label = record.get("category", "").strip()
            text = join_fields(record, text_fields)
            if label in label_to_id and text:
                self.examples.append(
                    {"text": text, "label": label_to_id[label], "language": record.get("_language")}
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        return self.examples[index]

    def label_ids(self) -> List[int]:
        return [example["label"] for example in self.examples]


class CategoryCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return dict(encoded)


class CategoryClassifier(nn.Module):
    """Encoder + dropout + linear head, with CLS or masked-mean pooling."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float = 0.1,
        pooling: str = "cls",
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling {pooling!r}")
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.accepts_token_type_ids = accepts_token_type_ids(encoder)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask) if self.pooling == "mean" else hidden[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def balanced_class_weights(dataset: CategoryDataset, num_labels: int) -> torch.Tensor:
    """``n_samples / (n_classes * count_c)``; zero for classes absent from training."""
    labels = dataset.label_ids()
    counts = Counter(labels)
    total = len(labels)
    return torch.tensor(
        [total / (num_labels * counts[i]) if counts[i] else 0.0 for i in range(num_labels)],
        dtype=torch.float,
    )


def label_diagnostics(
    label_to_id: Dict[str, int],
    train_records: Sequence[Dict],
    dev_records: Sequence[Dict],
    top_k: int = 20,
) -> Dict:
    train_counter = category_counter(train_records)
    dev_counter = category_counter(dev_records)
    unknown_dev = sorted(set(dev_counter) - set(label_to_id))
    return {
        "label_to_id": label_to_id,
        "raw_train_label_count": len(train_counter),
        "raw_dev_label_count": len(dev_counter),
        "unknown_dev_labels": unknown_dev,
        "unknown_dev_examples": sum(dev_counter[label] for label in unknown_dev),
        "train_distribution_top": train_counter.most_common(top_k),
        "dev_distribution_top": dev_counter.most_common(top_k),
    }


def classification_metrics(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def predict(model: CategoryClassifier, dataloader, device) -> Dict:
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = move_to(batch, device)
        outputs = model(**batch)
        total_loss += outputs["loss"].item()
        preds.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    model.train()
    return {"loss": total_loss / max(len(dataloader), 1), "preds": preds, "labels": labels}


def evaluate(model: CategoryClassifier, dataloader, device) -> Dict[str, float]:
    out = predict(model, dataloader, device)
    return {"loss": out["loss"], **classification_metrics(out["labels"], out["preds"])}


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps: int, max_grad_norm: float) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training"), start=1):
        batch = move_to(batch, device)
        loss = model(**batch)["loss"]
        (loss / grad_accum_steps).backward()
        total_loss += loss.item()
        if step % grad_accum_steps == 0 or step == len(dataloader):
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return total_loss / max(len(dataloader), 1)
