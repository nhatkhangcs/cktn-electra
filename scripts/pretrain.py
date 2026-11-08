#!/usr/bin/env python3
"""Continued pre-training of CKTN-ELECTRA (main model or an ablation).

Examples:
    python scripts/pretrain.py --config configs/pretrain/main.json
    python scripts/pretrain.py --config configs/pretrain/no_script_aware.json --batch_size 4
    python scripts/pretrain.py --config configs/pretrain/main.json --resume latest
"""

import argparse
from pathlib import Path

from cktn_electra.config import load_pretrain_config
from cktn_electra.pretrain import PretrainRun, run_pretraining
from cktn_electra.utils import REPO_ROOT, resolve_device, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pretrain/main.json")
    parser.add_argument("--output_dir", type=Path, default=None, help="Default: runs/pretrain/<variant>")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "runs/cache/chunks")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the config batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="'latest' or an epoch number.")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Debug: cap training steps per epoch.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Debug: cap evaluation batches.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_pretrain_config(args.config).with_overrides(batch_size=args.batch_size, seed=args.seed)
    run_pretraining(
        PretrainRun(
            config=config,
            output_dir=args.output_dir or REPO_ROOT / "runs/pretrain" / config.variant,
            corpus_dir=args.corpus_dir,
            cache_dir=args.cache_dir,
            device=resolve_device(args.device),
            num_workers=args.num_workers,
            resume=args.resume,
            rebuild_cache=args.rebuild_cache,
            max_steps_per_epoch=args.max_steps_per_epoch,
            max_eval_batches=args.max_eval_batches,
        )
    )


if __name__ == "__main__":
    main()
