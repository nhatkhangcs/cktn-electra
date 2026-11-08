#!/usr/bin/env python3
"""Generator MLM loss and perplexity on the development split.

Example:
    python scripts/eval_generator_ppl.py --run_dir runs/pretrain/main --checkpoint best
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cktn_electra.checkpoints import RunPaths, load_model_weights, resolve_tag
from cktn_electra.config import load_pretrain_config
from cktn_electra.data import ELECTRADataset, load_or_build_chunks, load_pretrain_tokenizer, split_files
from cktn_electra.modeling import CKTNElectra
from cktn_electra.utils import REPO_ROOT, resolve_device, save_json, set_seed, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pretrain/main.json")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", help="best, latest or an epoch number")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "runs/cache/chunks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    config = load_pretrain_config(args.config)
    paths = RunPaths(args.run_dir)
    tag = resolve_tag(paths, args.checkpoint)

    tokenizer = load_pretrain_tokenizer(config.base_model)
    chunks = load_or_build_chunks(
        "dev", split_files(args.corpus_dir, "dev"), tokenizer, config.seq_len, args.cache_dir, config.base_model
    )
    loader = DataLoader(
        ELECTRADataset(chunks, tokenizer, config.seq_len, config.mask_rate),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CKTNElectra(config, tokenizer=tokenizer, load_pretrained=False)
    load_model_weights(model, paths.disc_path(tag), paths.gen_path(tag))
    model.to(device).eval()

    total, batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generator MLM"):
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch, lam=0.0)["loss_mlm"].item()
            batches += 1
    loss = total / max(batches, 1)
    result = {
        "checkpoint": tag,
        "split": "dev",
        "generator_mlm_loss": loss,
        "generator_perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "num_batches": batches,
        "seed": args.seed,
    }
    save_json(paths.generator / f"generator_ppl_{tag}.json", result)
    print(result)


if __name__ == "__main__":
    main()
