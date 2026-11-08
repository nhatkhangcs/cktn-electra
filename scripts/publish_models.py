#!/usr/bin/env python3
"""Export pre-training discriminators to Hugging Face format and (optionally) upload them.

Every export is verified numerically against the raw checkpoint before upload.
The token is read from HF_TOKEN (or HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_HUB) in
the environment or in the repository's .env file.

Examples:
    # export + verify only
    python scripts/publish_models.py --only main
    # export, verify and upload everything in configs/release.json
    python scripts/publish_models.py --push
    # use checkpoints stored elsewhere
    python scripts/publish_models.py --push --checkpoint main=/path/to/discriminator_best.pt
"""

import argparse
from pathlib import Path

from cktn_electra.checkpoints import checkpoint_metadata
from cktn_electra.config import load_pretrain_config
from cktn_electra.hub import export_encoder, load_hf_token, push_folder
from cktn_electra.utils import REPO_ROOT, load_json, save_json, setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs/release.json")
    parser.add_argument("--only", nargs="+", default=None, help="Names from the manifest to process.")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH", help="Override a checkpoint path.")
    parser.add_argument("--export_root", type=Path, default=REPO_ROOT / "runs/hf_export")
    parser.add_argument("--push", action="store_true", help="Upload after a successful export.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--env_file", type=Path, default=REPO_ROOT / ".env")
    args = parser.parse_args()

    overrides = dict(item.split("=", 1) for item in args.checkpoint)
    entries = load_json(args.manifest)["models"]
    if args.only:
        entries = [e for e in entries if e["name"] in args.only]
    token = load_hf_token(args.env_file) if args.push else None

    summary = []
    for entry in entries:
        checkpoint = Path(overrides.get(entry["name"], REPO_ROOT / entry["checkpoint"]))
        config = load_pretrain_config(REPO_ROOT / entry["config"])
        card = (REPO_ROOT / entry["model_card"]).read_text(encoding="utf-8")
        export_dir = args.export_root / entry["repo_id"].split("/")[-1]

        result = export_encoder(checkpoint, config, export_dir, model_card=card)
        meta = checkpoint_metadata(checkpoint)
        save_json(export_dir / "pretraining_metrics.json", {"variant": config.variant, "config": config.to_dict(), **meta})
        result.update(name=entry["name"], checkpoint=str(checkpoint), epoch=meta["epoch"])
        if args.push:
            result["url"] = push_folder(
                export_dir, entry["repo_id"], token, private=args.private,
                commit_message=f"Upload CKTN-ELECTRA ({config.variant}) discriminator, epoch {meta['epoch']}",
            )
        summary.append(result)
        print(result)
    save_json(args.export_root / "publish_summary.json", summary)


if __name__ == "__main__":
    main()
