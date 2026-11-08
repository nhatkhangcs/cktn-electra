"""Export pre-training checkpoints to Hugging Face format and publish them."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from cktn_electra.checkpoints import build_hf_encoder
from cktn_electra.config import PretrainConfig
from cktn_electra.modeling import Discriminator, SharedEmbeddings
from cktn_electra.utils import torch_load

logger = logging.getLogger(__name__)

TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_HUB")


def load_hf_token(env_path: Optional[Path] = None) -> str:
    """Read a Hugging Face token from the environment or a ``.env`` file.

    Accepts ``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN`` or ``HUGGINGFACE_HUB``,
    quoted or unquoted.
    """
    for key in TOKEN_KEYS:
        if os.environ.get(key):
            return os.environ[key].strip()
    if env_path is not None and Path(env_path).exists():
        pattern = re.compile(r"^\s*(%s)\s*=\s*[\"']?([^\"'\s]+)[\"']?\s*$" % "|".join(TOKEN_KEYS))
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match:
                return match.group(2)
    raise KeyError(f"No Hugging Face token found. Set one of {TOKEN_KEYS} in the environment or in {env_path}.")


@torch.no_grad()
def verify_export(export_dir: Path, disc_checkpoint: Path, base_model: str, seq_len: int = 64, rtol: float = 1e-4) -> Dict:
    """Check that the exported encoder is the training-time discriminator.

    1. Every embedding/encoder tensor in the export must be bitwise equal to the checkpoint.
    2. The same random input through (a) ``SharedEmbeddings`` + ``Discriminator.encoder``
       built from the raw checkpoint and (b) ``AutoModel.from_pretrained(export_dir)``
       must agree up to floating-point noise (max |diff| <= rtol * max |hidden|);
       the two paths use different attention kernels, so exact equality is not expected.
    """
    config = AutoConfig.from_pretrained(base_model)
    state = torch_load(disc_checkpoint)
    exported = AutoModel.from_pretrained(export_dir).eval()

    exported_state = exported.state_dict()
    expected = {f"embeddings.{k}": v for k, v in state["shared_embeddings"].items()}
    expected.update({f"encoder.{k}": v for k, v in state["encoder"].items()})
    mismatched = [k for k, v in expected.items() if not torch.equal(exported_state[k], v)]
    if mismatched:
        raise RuntimeError(f"Exported weights differ from checkpoint: {mismatched[:5]}")

    embeddings = SharedEmbeddings(config)
    embeddings.load_state_dict(state["shared_embeddings"])
    discriminator = Discriminator(config)
    discriminator.encoder.load_state_dict(state["encoder"])
    embeddings.eval()
    discriminator.eval()
    del state

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(5, config.vocab_size, (2, seq_len), generator=generator)
    reference = discriminator.encoder(hidden_states=embeddings(input_ids), attention_mask=None)[0]
    candidate = exported(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).last_hidden_state
    max_diff = float((reference - candidate).abs().max())
    scale = float(reference.abs().max())
    if max_diff > rtol * scale:
        raise RuntimeError(f"Exported model output differs (max |diff| = {max_diff:.2e}, max |hidden| = {scale:.2e})")
    return {"tensors_checked": len(expected), "max_abs_diff": max_diff, "max_abs_hidden": scale}


def export_encoder(
    disc_checkpoint: Path,
    config: PretrainConfig,
    export_dir: Path,
    model_card: Optional[str] = None,
) -> Dict:
    """Write a ``RemBertModel`` + tokenizer (+ README) to ``export_dir`` and verify it."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = build_hf_encoder(disc_checkpoint, config.base_model)
    model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)
    tokenizer.save_pretrained(export_dir)
    _copy_sentencepiece_model(config.base_model, export_dir)
    if model_card is not None:
        (export_dir / "README.md").write_text(model_card, encoding="utf-8")
    del model

    check = verify_export(export_dir, disc_checkpoint, config.base_model)
    logger.info(
        "Exported %s -> %s (%d tensors bitwise equal; forward max |diff| %.1e of max |hidden| %.1e)",
        disc_checkpoint, export_dir, check["tensors_checked"], check["max_abs_diff"], check["max_abs_hidden"],
    )
    return {"export_dir": str(export_dir), **check}


def _copy_sentencepiece_model(base_model: str, export_dir: Path) -> None:
    """Ship the raw SentencePiece model too, so older ``transformers`` can build the slow tokenizer."""
    import shutil

    from huggingface_hub import hf_hub_download

    local = Path(base_model) / "sentencepiece.model"
    try:
        source = local if local.exists() else Path(hf_hub_download(base_model, "sentencepiece.model"))
    except Exception as error:  # noqa: BLE001 - optional file
        logger.warning("No sentencepiece.model found for %s (%s)", base_model, error)
        return
    shutil.copyfile(source, export_dir / "sentencepiece.model")


def push_folder(export_dir: Path, repo_id: str, token: str, private: bool = False, commit_message: str = "Upload model") -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=repo_id, repo_type="model", commit_message=commit_message)
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Uploaded %s to %s", export_dir, url)
    return url
