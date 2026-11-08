#!/usr/bin/env python3
"""Tokenise the CKTN corpus once and cache the 512-token chunks used for pre-training.

Example:
    python scripts/prepare_cache.py --config configs/pretrain/main.json
"""

import argparse
from pathlib import Path

from cktn_electra.config import load_pretrain_config
from cktn_electra.data import load_pretrain_tokenizer
from cktn_electra.pretrain import build_chunks
from cktn_electra.utils import REPO_ROOT, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pretrain/main.json")
    parser.add_argument("--corpus_dir", type=Path, default=REPO_ROOT / "corpus")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / "runs/cache/chunks")
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()

    config = load_pretrain_config(args.config)
    tokenizer = load_pretrain_tokenizer(config.base_model)
    chunks = build_chunks(config, tokenizer, args.corpus_dir, args.cache_dir, args.rebuild_cache)
    print({split: len(items) for split, items in chunks.items()})


if __name__ == "__main__":
    main()
