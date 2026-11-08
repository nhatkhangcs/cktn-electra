"""Checkpoint I/O for pre-training runs and conversion to Hugging Face ``RemBertModel``.

Run directory layout (also used by checkpoints from earlier versions of this code)::

    <run_dir>/checkpoint/
        discriminator/discriminator_epoch{N}.pt, discriminator_best.pt
        generator/generator_epoch{N}.pt,         generator_best.pt
        state/training_state_epoch{N}.pt,        training_state_latest.pt
        graphs/*.png
        report.json
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, RemBertConfig, RemBertModel

from cktn_electra.modeling import CKTNElectra
from cktn_electra.utils import torch_load

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint"

    @property
    def discriminator(self) -> Path:
        return self.checkpoint / "discriminator"

    @property
    def generator(self) -> Path:
        return self.checkpoint / "generator"

    @property
    def state(self) -> Path:
        return self.checkpoint / "state"

    @property
    def graphs(self) -> Path:
        return self.checkpoint / "graphs"

    @property
    def report(self) -> Path:
        return self.checkpoint / "report.json"

    def make_dirs(self) -> None:
        for path in (self.discriminator, self.generator, self.state, self.graphs):
            path.mkdir(parents=True, exist_ok=True)

    def disc_path(self, tag: str) -> Path:
        return self.discriminator / f"discriminator_{tag}.pt"

    def gen_path(self, tag: str) -> Path:
        return self.generator / f"generator_{tag}.pt"

    def state_path(self, tag: str) -> Path:
        return self.state / f"training_state_{tag}.pt"


def _epoch_of(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{prefix}_epoch(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def available_epochs(paths: RunPaths) -> List[int]:
    """Epochs for which both a discriminator and a generator checkpoint exist."""
    disc = {_epoch_of(p, "discriminator") for p in paths.discriminator.glob("discriminator_epoch*.pt")}
    gen = {_epoch_of(p, "generator") for p in paths.generator.glob("generator_epoch*.pt")}
    return sorted(epoch for epoch in disc & gen if epoch > 0)


def resolve_tag(paths: RunPaths, request: str) -> str:
    """Map ``best`` / ``latest`` / ``<epoch>`` to a checkpoint tag such as ``epoch4``."""
    if request == "best":
        return "best"
    epochs = available_epochs(paths)
    if not epochs:
        raise FileNotFoundError(f"No epoch checkpoints under {paths.checkpoint}")
    if request == "latest":
        return f"epoch{epochs[-1]}"
    if request.isdigit() and int(request) in epochs:
        return f"epoch{int(request)}"
    raise ValueError(f"Unknown checkpoint {request!r}; available epochs: {epochs}")


def save_epoch(
    paths: RunPaths,
    model: CKTNElectra,
    epoch: int,
    metrics: Dict,
    is_best: bool,
    training_state: Dict,
) -> None:
    disc_state = {
        "encoder": model.discriminator.encoder.state_dict(),
        "rtd_head": model.discriminator.rtd_head.state_dict(),
        "shared_embeddings": model.shared_embeddings.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    gen_state = {"generator": model.generator.state_dict(), "epoch": epoch, "metrics": metrics}
    tag = f"epoch{epoch}"
    torch.save(disc_state, paths.disc_path(tag))
    torch.save(gen_state, paths.gen_path(tag))
    torch.save(training_state, paths.state_path(tag))
    torch.save(training_state, paths.state_path("latest"))
    if is_best:
        torch.save(disc_state, paths.disc_path("best"))
        torch.save(gen_state, paths.gen_path("best"))
        logger.info("New best checkpoint (RTD F1=%.4f)", metrics["f1"])
    logger.info("Saved epoch %d checkpoints to %s", epoch, paths.checkpoint)


def load_model_weights(
    model: CKTNElectra, disc_path: Path, gen_path: Optional[Path] = None
) -> Tuple[int, int]:
    """Load discriminator (+ shared embeddings) and optionally generator weights.

    Returns the epochs recorded in the discriminator and generator checkpoints
    (``-1`` if absent).
    """
    disc_state = torch_load(disc_path)
    model.discriminator.encoder.load_state_dict(disc_state["encoder"])
    model.discriminator.rtd_head.load_state_dict(disc_state["rtd_head"])
    model.shared_embeddings.load_state_dict(disc_state["shared_embeddings"])
    disc_epoch = int(disc_state.get("epoch", -1))
    del disc_state

    gen_epoch = -1
    if gen_path is not None:
        gen_state = torch_load(gen_path)
        model.generator.load_state_dict(gen_state["generator"])
        gen_epoch = int(gen_state.get("epoch", -1))
        del gen_state
    return disc_epoch, gen_epoch


def build_hf_encoder(
    disc_checkpoint: Path, base_model: str, config: Optional[RemBertConfig] = None
) -> RemBertModel:
    """Rebuild a Hugging Face ``RemBertModel`` from a pre-training discriminator checkpoint.

    The shared embeddings become ``model.embeddings`` and the discriminator
    encoder becomes ``model.encoder``. The RTD head is dropped; the pooler is
    left at its initialisation (it is not trained during pre-training).
    """
    disc_checkpoint = Path(disc_checkpoint)
    if not disc_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {disc_checkpoint}")
    config = config or AutoConfig.from_pretrained(base_model)
    model = RemBertModel(config)
    state = torch_load(disc_checkpoint)

    shared = state["shared_embeddings"]
    emb = model.embeddings
    emb.word_embeddings.weight.data.copy_(shared["word_embeddings.weight"])
    emb.position_embeddings.weight.data.copy_(shared["position_embeddings.weight"])
    emb.token_type_embeddings.weight.data.copy_(shared["token_type_embeddings.weight"])
    emb.LayerNorm.weight.data.copy_(shared["LayerNorm.weight"])
    emb.LayerNorm.bias.data.copy_(shared["LayerNorm.bias"])
    model.encoder.load_state_dict(state["encoder"])
    model.eval()
    return model


def checkpoint_metadata(disc_checkpoint: Path) -> Dict:
    state = torch_load(disc_checkpoint, mmap=True)
    return {"epoch": state.get("epoch"), "metrics": state.get("metrics", {})}
