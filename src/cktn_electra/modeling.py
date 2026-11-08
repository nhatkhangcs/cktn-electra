"""CKTN-ELECTRA architecture.

* ``SharedEmbeddings``: token, position and segment embeddings shared by the
  generator and the discriminator, updated by both losses.
* ``Generator``: a small Transformer, 1/4 of the discriminator depth, trained
  from scratch with masked language modelling.
* ``DifficultyCalibratedSampler``: draws replacements from the generator's
  filtered top-k distribution.
* ``Discriminator``: a 32-layer RemBERT encoder with a replaced-token-detection
  head, initialised from the vocabulary-augmented RemBERT checkpoint.

Training objective: ``L = L_MLM + lambda(t) * L_RTD``.

Module attribute names (``shared_embeddings``, ``generator``,
``discriminator.encoder``, ``discriminator.rtd_head``) are part of the
checkpoint format; do not rename them.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase, RemBertConfig
from transformers.models.rembert.modeling_rembert import RemBertEncoder

from cktn_electra.config import PretrainConfig, SamplerConfig
from cktn_electra.text import SCRIPT_SHARED, is_noise_surface, script_profile, token_surface

logger = logging.getLogger(__name__)


class SharedEmbeddings(nn.Module):
    """Input embeddings shared by the generator and the discriminator.

    RemBERT decouples input and output embeddings, so only the input side
    (E_token, E_pos, segment embeddings, LayerNorm) is shared. The generator's
    output projection is separate, as in standard ELECTRA.
    """

    def __init__(self, config: RemBertConfig):
        super().__init__()
        dim = config.input_embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size, dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, dim)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = self.position_ids[:, : input_ids.size(1)]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedded = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.LayerNorm(embedded))


def make_generator_config(disc_config: RemBertConfig, depth_divisor: int = 4) -> RemBertConfig:
    """Generator config: same width/heads/vocab as the discriminator, fewer layers."""
    gen_config = copy.deepcopy(disc_config)
    gen_config.num_hidden_layers = max(1, disc_config.num_hidden_layers // depth_divisor)
    return gen_config


class GeneratorMLMHead(nn.Module):
    """MLM prediction head; the output projection is not tied to the embeddings."""

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.input_embedding_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.input_embedding_size, config.vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))


class Generator(nn.Module):
    """Small Transformer encoder trained from scratch on top of the shared embeddings."""

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.config = config
        self.embeddings_project = nn.Linear(config.input_embedding_size, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.mlm_head = GeneratorMLMHead(config)

    def forward(
        self, embedded: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Map shared embeddings ``[B, L, E]`` to MLM logits ``[B, L, V]``."""
        hidden = self.embeddings_project(embedded)
        pad_mask = attention_mask == 0 if attention_mask is not None else None
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        return self.mlm_head(hidden)


class DiscriminatorRTDHead(nn.Module):
    """Per-token binary head: logit > 0 means *original*, < 0 means *replaced*."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.dense(hidden_states)))


class Discriminator(nn.Module):
    """RemBERT encoder plus RTD head.

    ``RemBertEncoder`` contains its own ``embedding_hidden_mapping_in``
    projection (input_embedding_size -> hidden_size), so the shared embeddings
    are passed in directly.
    """

    def __init__(self, config: RemBertConfig):
        super().__init__()
        self.config = config
        self.encoder = RemBertEncoder(config)
        self.rtd_head = DiscriminatorRTDHead(config.hidden_size)

    def forward(
        self, embedded: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return per-token RTD logits ``[B, L, 1]``."""
        ext_mask = None
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0
        hidden = self.encoder(hidden_states=embedded, attention_mask=ext_mask)[0]
        return self.rtd_head(hidden)


class DifficultyCalibratedSampler(nn.Module):
    """Draw generator replacements from a filtered top-k band.

    For every masked position, candidates are restricted to the generator's
    top-k logits and must satisfy four constraints:

    1. differ from the original token;
    2. be well formed: not a special token, and not a noise token
       (punctuation/digit/symbol-dominant) unless the original is one too;
    3. be script compatible: same script class as the original, or
       script-neutral (disabled when ``script_aware=False``);
    4. have embedding cosine similarity to the original in [rho_min, rho_max].

    A replacement is sampled from a temperature-scaled softmax over the
    surviving candidates. If no candidate survives, the token is left
    unchanged.
    """

    def __init__(
        self,
        vocab_size: int,
        sampler_config: SamplerConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        super().__init__()
        self.config = sampler_config

        token_is_special = torch.zeros(vocab_size, dtype=torch.bool)
        token_is_noise = torch.zeros(vocab_size, dtype=torch.bool)
        token_script = torch.full((vocab_size,), SCRIPT_SHARED, dtype=torch.long)

        if tokenizer is not None:
            special_ids = {
                tid for tid in tokenizer.all_special_ids if tid is not None and 0 <= tid < vocab_size
            }
            tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
            for token_id, token in enumerate(tokens):
                surface = token_surface(str(token))
                token_is_special[token_id] = token_id in special_ids
                token_is_noise[token_id] = is_noise_surface(surface)
                token_script[token_id] = script_profile(surface)
        else:
            logger.warning(
                "DifficultyCalibratedSampler built without a tokenizer: special-token, "
                "noise and script filters are inactive."
            )

        self.register_buffer("token_is_special", token_is_special, persistent=False)
        self.register_buffer("token_is_noise", token_is_noise, persistent=False)
        self.register_buffer("token_script", token_script, persistent=False)

    def script_compatible(
        self, candidate_script: torch.Tensor, original_script: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.script_aware:
            return torch.ones_like(candidate_script, dtype=torch.bool)
        return (candidate_script == original_script) | (candidate_script == SCRIPT_SHARED)

    @torch.no_grad()
    def forward(
        self,
        gen_logits: torch.Tensor,
        original_ids: torch.Tensor,
        is_masked: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        embedding_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        top_k = min(self.config.top_k, gen_logits.size(-1))
        top_logits, top_ids = torch.topk(gen_logits, k=top_k, dim=-1)

        active_mask = is_masked
        if attention_mask is not None:
            active_mask = active_mask & attention_mask.bool()

        # (1) identity and (2) special-token constraints
        valid = top_ids != original_ids.unsqueeze(-1)
        valid &= ~self.token_is_special[top_ids]

        # (2) noise tokens may only replace noise tokens
        original_noise = self.token_is_noise[original_ids].unsqueeze(-1)
        valid &= ~self.token_is_noise[top_ids] | original_noise

        # (3) script compatibility
        original_script = self.token_script[original_ids].unsqueeze(-1)
        valid &= self.script_compatible(self.token_script[top_ids], original_script)

        # (4) embedding-similarity band
        original_emb = F.embedding(original_ids, embedding_weight)
        candidate_emb = F.embedding(top_ids, embedding_weight)
        similarity = F.cosine_similarity(candidate_emb, original_emb.unsqueeze(-2), dim=-1)
        valid &= (similarity >= self.config.rho_min) & (similarity <= self.config.rho_max)
        valid &= active_mask.unsqueeze(-1)

        temperature = max(self.config.temperature, 1e-6)
        calibrated_logits = (top_logits / temperature).masked_fill(~valid, -1e9)
        probs = torch.softmax(calibrated_logits, dim=-1)
        sampled_index = torch.multinomial(probs.reshape(-1, top_k), num_samples=1).view(
            original_ids.shape
        )

        sampled_ids = top_ids.gather(-1, sampled_index.unsqueeze(-1)).squeeze(-1)
        sampled_valid = valid.gather(-1, sampled_index.unsqueeze(-1)).squeeze(-1)
        replacement_mask = sampled_valid & (sampled_ids != original_ids)
        corrupted_ids = torch.where(replacement_mask, sampled_ids, original_ids)

        active_count = active_mask.float().sum().clamp_min(1.0)
        return {
            "corrupted_ids": corrupted_ids,
            "replacement_mask": replacement_mask,
            "replacement_rate": replacement_mask.float().sum() / active_count,
            "valid_candidate_rate": (valid.any(dim=-1) & active_mask).float().sum() / active_count,
        }


class CKTNElectra(nn.Module):
    """Generator + calibrated sampler + discriminator with shared embeddings.

    Args:
        config: pre-training configuration (sampler settings, generator depth,
            base checkpoint).
        tokenizer: tokenizer of ``config.base_model``; required for the
            sampler's special/noise/script filters.
        load_pretrained: copy the base checkpoint's embeddings and encoder into
            the shared embeddings and discriminator.
        disc_config: optional discriminator config; when omitted it is fetched
            from ``config.base_model``. Pass a small config for offline tests.
    """

    def __init__(
        self,
        config: Optional[PretrainConfig] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        load_pretrained: bool = True,
        disc_config: Optional[RemBertConfig] = None,
    ):
        super().__init__()
        self.config = config or PretrainConfig()
        if disc_config is None:
            disc_config = AutoConfig.from_pretrained(self.config.base_model)
        self.disc_config = disc_config

        gen_config = make_generator_config(disc_config, self.config.generator_depth_divisor)
        logger.info(
            "Discriminator: %d layers, hidden=%d, vocab=%d | generator: %d layers",
            disc_config.num_hidden_layers,
            disc_config.hidden_size,
            disc_config.vocab_size,
            gen_config.num_hidden_layers,
        )

        self.shared_embeddings = SharedEmbeddings(disc_config)
        self.generator = Generator(gen_config)
        self.discriminator = Discriminator(disc_config)
        self.replacement_sampler = DifficultyCalibratedSampler(
            disc_config.vocab_size, self.config.sampler, tokenizer=tokenizer
        )

        if load_pretrained:
            self.load_base_weights(self.config.base_model)

    def load_base_weights(self, model_name_or_path: str) -> None:
        """Initialise shared embeddings and discriminator encoder from a RemBERT checkpoint.

        The generator keeps its random initialisation.
        """
        logger.info("Loading base weights from %s", model_name_or_path)
        base = AutoModel.from_pretrained(model_name_or_path)
        src = base.embeddings
        dst = self.shared_embeddings
        dst.word_embeddings.weight.data.copy_(src.word_embeddings.weight.data)
        dst.position_embeddings.weight.data.copy_(src.position_embeddings.weight.data)
        dst.token_type_embeddings.weight.data.copy_(src.token_type_embeddings.weight.data)
        dst.LayerNorm.weight.data.copy_(src.LayerNorm.weight.data)
        dst.LayerNorm.bias.data.copy_(src.LayerNorm.bias.data)
        self.discriminator.encoder.load_state_dict(base.encoder.state_dict())
        del base

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        lam: float = 0.0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Run one generator/discriminator step.

        Args:
            input_ids: masked input ids ``[B, L]``.
            attention_mask: 1 for real tokens, 0 for padding.
            token_type_ids: segment ids (all zeros in pre-training).
            labels: original ids at masked positions, -100 elsewhere.
            lam: current discriminator loss weight; the discriminator is
                skipped entirely when ``lam == 0``.
        """
        device = input_ids.device

        # 1. Generator MLM on the masked input
        gen_logits = self.generator(
            self.shared_embeddings(input_ids, token_type_ids=token_type_ids), attention_mask
        )
        loss_mlm = torch.tensor(0.0, device=device)
        if labels is not None:
            loss_mlm = F.cross_entropy(
                gen_logits.reshape(-1, gen_logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
            is_masked = labels != -100
            original_ids = torch.where(is_masked, labels, input_ids)
        else:
            is_masked = torch.zeros_like(input_ids, dtype=torch.bool)
            original_ids = input_ids

        # 2. Difficulty-calibrated replacement sampling
        sample = self.replacement_sampler(
            gen_logits=gen_logits,
            original_ids=original_ids,
            is_masked=is_masked,
            attention_mask=attention_mask,
            embedding_weight=self.shared_embeddings.word_embeddings.weight,
        )
        replacement_mask = sample["replacement_mask"]

        # RTD labels are relative to the original token: 1 = original, 0 = replaced.
        rtd_labels = (~replacement_mask).float()
        rtd_label_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )

        loss_disc = torch.tensor(0.0, device=device)
        disc_logits = None
        avg_disc_confidence = torch.tensor(0.0, device=device)
        rtd_entropy = torch.tensor(0.0, device=device)

        # 3. Discriminator RTD over every non-padding token
        if lam > 0.0:
            embedded_corrupted = self.shared_embeddings(
                sample["corrupted_ids"], token_type_ids=token_type_ids
            )
            disc_logits = self.discriminator(embedded_corrupted, attention_mask)
            token_mask = rtd_label_mask.float()
            active_count = token_mask.sum().clamp_min(1.0)

            per_token = F.binary_cross_entropy_with_logits(
                disc_logits.squeeze(-1), rtd_labels, reduction="none"
            )
            loss_disc = (per_token * token_mask).sum() / active_count

            probs = torch.sigmoid(disc_logits.squeeze(-1)).clamp(1e-6, 1.0 - 1e-6)
            confidence = torch.maximum(probs, 1.0 - probs)
            entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
            avg_disc_confidence = (confidence * token_mask).sum() / active_count
            rtd_entropy = (entropy * token_mask).sum() / active_count

        return {
            "loss": loss_mlm + lam * loss_disc,
            "loss_mlm": loss_mlm,
            "loss_disc": loss_disc,
            "gen_logits": gen_logits,
            "disc_logits": disc_logits,
            "corrupted_ids": sample["corrupted_ids"],
            "replacement_mask": replacement_mask,
            "rtd_labels": rtd_labels,
            "rtd_label_mask": rtd_label_mask,
            "replacement_rate": sample["replacement_rate"],
            "valid_candidate_rate": sample["valid_candidate_rate"],
            "avg_disc_confidence": avg_disc_confidence,
            "rtd_entropy": rtd_entropy,
        }


_NO_DECAY_TERMS = ("bias", "LayerNorm.weight", "layer_norm.weight")


def _split_decay(module: nn.Module):
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if any(term in name for term in _NO_DECAY_TERMS) else decay).append(param)
    return decay, no_decay


def decay_parameter_groups(module: nn.Module, weight_decay: float):
    """Two AdamW groups: weight decay everywhere except biases and LayerNorm weights."""
    decay, no_decay = _split_decay(module)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def pretrain_parameter_groups(model: CKTNElectra, weight_decay: float):
    """AdamW groups for pre-training.

    Parameters are ordered discriminator -> generator -> shared embeddings so
    that optimizer states saved by earlier runs can be resumed.
    """
    parts = [_split_decay(m) for m in (model.discriminator, model.generator, model.shared_embeddings)]
    return [
        {"params": [p for d, _ in parts for p in d], "weight_decay": weight_decay},
        {"params": [p for _, n in parts for p in n], "weight_decay": 0.0},
    ]
