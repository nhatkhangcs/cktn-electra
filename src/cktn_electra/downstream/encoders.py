"""Encoder/tokenizer loading shared by the downstream tasks.

An encoder is chosen either by a benchmark key (``--model_key``) or by a local
CKTN-ELECTRA pre-training checkpoint (``--checkpoint``), which is converted to
a ``RemBertModel`` on the fly.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from cktn_electra.checkpoints import build_hf_encoder
from cktn_electra.config import VOCAB_AUGMENTED_REMBERT

RELEASED_MODEL = "nhatkhangdtp/CKTN-ELECTRA"

BENCHMARK_MODELS: Dict[str, str] = {
    "cktn": RELEASED_MODEL,  # CKTN-ELECTRA after continued pre-training
    "cktn_init": VOCAB_AUGMENTED_REMBERT,  # vocabulary-augmented RemBERT, before pre-training
    "mbert": "google-bert/bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "rembert": "google/rembert",
}

# RemBERT-family models were trained with the SentencePiece (slow) tokenizer.
_SLOW_TOKENIZER_KEYS = {"cktn", "cktn_init", "rembert"}


def resolve_model_id(model_key: str, model_name_or_path: Optional[str] = None) -> str:
    if model_name_or_path:
        return str(model_name_or_path)
    if model_key not in BENCHMARK_MODELS:
        raise KeyError(f"Unknown model key {model_key!r}; choose from {sorted(BENCHMARK_MODELS)}")
    return BENCHMARK_MODELS[model_key]


def load_tokenizer(model_key: str, model_name_or_path: Optional[str] = None):
    use_fast = model_key not in _SLOW_TOKENIZER_KEYS
    return AutoTokenizer.from_pretrained(resolve_model_id(model_key, model_name_or_path), use_fast=use_fast)


def load_encoder(
    model_key: str,
    model_name_or_path: Optional[str] = None,
    checkpoint: Optional[Path] = None,
) -> Tuple[nn.Module, str]:
    """Return ``(encoder, description)``.

    With ``checkpoint`` set, a CKTN-ELECTRA pre-training discriminator ``.pt``
    is rebuilt on top of the config of ``model_name_or_path`` (or the model
    key's Hub id).
    """
    model_id = resolve_model_id(model_key, model_name_or_path)
    if checkpoint is not None:
        return build_hf_encoder(Path(checkpoint), model_id), f"{checkpoint} (config: {model_id})"
    return AutoModel.from_pretrained(model_id), model_id


def empty_encoder(model_key: str, model_name_or_path: Optional[str] = None) -> nn.Module:
    """Randomly initialised encoder with the right architecture, for loading fine-tuned weights."""
    return AutoModel.from_config(AutoConfig.from_pretrained(resolve_model_id(model_key, model_name_or_path)))


def accepts_token_type_ids(encoder: nn.Module) -> bool:
    return "token_type_ids" in inspect.signature(encoder.forward).parameters


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


def move_to(batch, device):
    """Recursively move tensors in (nested) dicts to ``device``."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return {key: move_to(value, device) for key, value in batch.items()}
