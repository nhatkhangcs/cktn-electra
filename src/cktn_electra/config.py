"""Configuration objects for CKTN-ELECTRA continued pre-training.

The defaults reproduce the full CKTN-ELECTRA model described in the paper.
Each ablation is a JSON file under ``configs/pretrain/`` that overrides only
the fields it changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

VOCAB_AUGMENTED_REMBERT = "ducanhdinh/CKTN-ELECTRA"
ORIGINAL_REMBERT = "google/rembert"

LambdaSchedule = Literal["linear", "constant"]


@dataclass(frozen=True)
class SamplerConfig:
    """Difficulty-calibrated replacement sampler (Section 3.3 of the paper)."""

    top_k: int = 64
    temperature: float = 1.25
    rho_min: float = 0.15
    rho_max: float = 0.95
    # When False, the script-compatibility filter is disabled
    # (the "no script-aware" ablation).
    script_aware: bool = True


@dataclass(frozen=True)
class PretrainConfig:
    """Hyper-parameters for one continued pre-training run."""

    variant: str = "main"
    # Discriminator initialisation, tokenizer and shared embeddings.
    base_model: str = VOCAB_AUGMENTED_REMBERT
    # Generator depth = discriminator depth // generator_depth_divisor.
    generator_depth_divisor: int = 4

    total_epochs: int = 6
    mask_rate: float = 0.15
    seq_len: int = 512
    batch_size: int = 8
    lr: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_norm: float = 1.0
    seed: Optional[int] = 42

    # Generator/discriminator loss weighting: L = L_MLM + lambda(t) * L_RTD.
    lambda_max: float = 50.0
    lambda_schedule: LambdaSchedule = "linear"
    lambda_zero_until_epoch: int = 2
    lambda_ramp_until_epoch: int = 3

    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    def __post_init__(self) -> None:
        if self.lambda_schedule not in ("linear", "constant"):
            raise ValueError(f"Unknown lambda_schedule: {self.lambda_schedule!r}")
        if self.generator_depth_divisor < 1:
            raise ValueError("generator_depth_divisor must be >= 1")
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError("mask_rate must be in (0, 1)")
        if self.lambda_ramp_until_epoch < self.lambda_zero_until_epoch:
            raise ValueError("lambda_ramp_until_epoch must be >= lambda_zero_until_epoch")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "PretrainConfig":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PretrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known - {"description"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        data = {k: v for k, v in payload.items() if k in known}
        if "sampler" in data and not isinstance(data["sampler"], SamplerConfig):
            data["sampler"] = SamplerConfig(**data["sampler"])
        return cls(**data)


def load_pretrain_config(path: Union[str, Path]) -> PretrainConfig:
    """Load a :class:`PretrainConfig` from a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return PretrainConfig.from_dict(json.load(handle))
