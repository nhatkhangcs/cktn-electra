"""Continued pre-training loop for CKTN-ELECTRA."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from cktn_electra.checkpoints import RunPaths, available_epochs, load_model_weights, resolve_tag, save_epoch
from cktn_electra.config import PretrainConfig
from cktn_electra.data import ELECTRADataset, load_or_build_chunks, load_pretrain_tokenizer, split_files
from cktn_electra.modeling import CKTNElectra, pretrain_parameter_groups
from cktn_electra.schedule import build_lambda_scheduler
from cktn_electra.utils import load_json, save_json, seed_worker, set_seed, torch_load

logger = logging.getLogger(__name__)

REPORT_KEYS = (
    "avg_mlm_loss",
    "avg_rtd_loss",
    "accuracy",
    "f1",
    "replacement_rate",
    "valid_candidate_rate",
    "disc_confidence",
    "rtd_entropy",
    "grad_norm_ratio",
)


@dataclass
class PretrainRun:
    config: PretrainConfig
    output_dir: Path
    corpus_dir: Path
    cache_dir: Path
    device: torch.device
    num_workers: int = 4
    resume: Optional[str] = None
    rebuild_cache: bool = False
    max_steps_per_epoch: Optional[int] = None  # for smoke tests only
    max_eval_batches: Optional[int] = None  # for smoke tests only


def build_chunks(config: PretrainConfig, tokenizer, corpus_dir: Path, cache_dir: Path, rebuild: bool):
    return {
        split: load_or_build_chunks(
            split,
            split_files(corpus_dir, split),
            tokenizer,
            config.seq_len,
            cache_dir,
            config.base_model,
            rebuild,
        )
        for split in ("train", "dev")
    }


@torch.no_grad()
def evaluate_rtd(
    model: CKTNElectra,
    dataloader: DataLoader,
    device: torch.device,
    lam: float,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """MLM/RTD losses and replacement diagnostics on freshly sampled corruptions.

    RTD F1 treats *replaced* (label 0) as the positive class.
    """
    model.eval()
    sums = {key: 0.0 for key in ("mlm", "rtd", "rep", "valid", "conf", "ent")}
    preds, golds = [], []
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if max_batches is not None and n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, lam=lam)
        sums["mlm"] += out["loss_mlm"].item()
        sums["rtd"] += out["loss_disc"].item()
        sums["rep"] += out["replacement_rate"].item()
        sums["valid"] += out["valid_candidate_rate"].item()
        sums["conf"] += out["avg_disc_confidence"].item()
        sums["ent"] += out["rtd_entropy"].item()
        n_batches += 1

        if out["disc_logits"] is not None:
            mask = out["rtd_label_mask"]
            if mask.any():
                pred = (torch.sigmoid(out["disc_logits"].squeeze(-1)) > 0.5).long()
                preds.append(pred[mask].cpu().numpy())
                golds.append(out["rtd_labels"][mask].long().cpu().numpy())

    model.train()
    n = max(n_batches, 1)
    if preds:
        y_pred, y_true = np.concatenate(preds), np.concatenate(golds)
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    else:
        acc, f1 = 0.0, 0.0
    return {
        "avg_mlm_loss": sums["mlm"] / n,
        "avg_rtd_loss": sums["rtd"] / n,
        "accuracy": acc,
        "f1": f1,
        "replacement_rate": sums["rep"] / n,
        "valid_candidate_rate": sums["valid"] / n,
        "disc_confidence": sums["conf"] / n,
        "rtd_entropy": sums["ent"] / n,
    }


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += param.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


def generator_discriminator_grad_ratio(model: CKTNElectra) -> float:
    disc = _grad_norm(model.discriminator)
    return _grad_norm(model.generator) / disc if disc > 0.0 else 0.0


def _set_scheduler_step(lr_scheduler, optimizer, steps: int) -> None:
    """Fast-forward a LambdaLR schedule when resuming without a saved scheduler state."""
    if steps <= 0:
        return
    lr_scheduler.last_epoch = steps
    lrs = [base * fn(steps) for base, fn in zip(lr_scheduler.base_lrs, lr_scheduler.lr_lambdas)]
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr
    lr_scheduler._last_lr = lrs


def plot_report(report: List[Dict], graphs_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "avg_mlm_loss": ("MLM loss", "Loss"),
        "avg_rtd_loss": ("RTD loss", "Loss"),
        "accuracy": ("RTD accuracy", "Accuracy"),
        "f1": ("RTD replaced-token F1", "F1"),
        "replacement_rate": ("Replacement rate", "Rate"),
        "valid_candidate_rate": ("Valid-candidate rate", "Rate"),
        "disc_confidence": ("Discriminator confidence", "Confidence"),
        "rtd_entropy": ("RTD prediction entropy", "Entropy"),
        "grad_norm_ratio": ("Generator / discriminator grad-norm ratio", "Ratio"),
    }
    graphs_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in report]
    n_cols = 3
    n_rows = math.ceil(len(titles) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    for ax, (key, (title, ylabel)) in zip(axes.flat, titles.items()):
        ax.plot(epochs, [row[key] for row in report], marker="o", linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(True, alpha=0.4)
    for ax in axes.flat[len(titles):]:
        ax.axis("off")
    fig.suptitle("CKTN-ELECTRA training metrics")
    fig.tight_layout()
    fig.savefig(graphs_dir / "training_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training graphs to %s", graphs_dir)


def run_pretraining(run: PretrainRun) -> List[Dict]:
    config = run.config
    if config.seed is not None:
        set_seed(config.seed)
    paths = RunPaths(Path(run.output_dir))
    paths.make_dirs()
    save_json(paths.checkpoint / "config.json", config.to_dict())

    tokenizer = load_pretrain_tokenizer(config.base_model)
    chunks = build_chunks(config, tokenizer, run.corpus_dir, run.cache_dir, run.rebuild_cache)
    pin = run.device.type == "cuda"
    train_loader = DataLoader(
        ELECTRADataset(chunks["train"], tokenizer, config.seq_len, config.mask_rate),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker,
    )
    dev_loader = DataLoader(
        ELECTRADataset(chunks["dev"], tokenizer, config.seq_len, config.mask_rate),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=run.num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker,
    )

    model = CKTNElectra(config, tokenizer=tokenizer, load_pretrained=True).to(run.device)
    optimizer = torch.optim.AdamW(pretrain_parameter_groups(model, config.weight_decay), lr=config.lr)

    steps_per_epoch = len(train_loader)
    if run.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, run.max_steps_per_epoch)
    total_steps = config.total_epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    lambda_scheduler = build_lambda_scheduler(config, steps_per_epoch)
    logger.info(
        "steps/epoch=%d total=%d warmup=%d lambda=%s(max=%.1f)",
        steps_per_epoch, total_steps, warmup_steps, config.lambda_schedule, config.lambda_max,
    )

    report: List[Dict] = []
    best_f1, global_step, start_epoch = -1.0, 0, 1
    if run.resume is not None:
        tag = resolve_tag(paths, run.resume)
        if tag == "best":
            raise ValueError("Resume from 'latest' or an epoch number, not 'best'.")
        epoch = int(tag.removeprefix("epoch"))
        load_model_weights(model, paths.disc_path(tag), paths.gen_path(tag))
        state_path = paths.state_path(tag)
        if state_path.exists():
            state = torch_load(state_path, map_location=str(run.device))
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            global_step, best_f1, report = state["global_step"], state["best_f1"], state["report"]
        else:
            report = [row for row in load_json(paths.report) if row["epoch"] <= epoch] if paths.report.exists() else []
            best_f1 = max((row["f1"] for row in report), default=-1.0)
            global_step = epoch * steps_per_epoch
            _set_scheduler_step(lr_scheduler, optimizer, global_step)
        start_epoch = epoch + 1
        logger.info("Resuming at epoch %d (global_step=%d, available=%s)", start_epoch, global_step, available_epochs(paths))

    model.train()
    for epoch in range(start_epoch, config.total_epochs + 1):
        sums = {"mlm": 0.0, "rtd": 0.0, "rep": 0.0, "ratio": 0.0}
        n_batches = 0
        progress = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch}/{config.total_epochs}")
        for batch in progress:
            if n_batches >= steps_per_epoch:
                break
            batch = {k: v.to(run.device) for k, v in batch.items()}
            lam = lambda_scheduler.get_lambda(global_step)
            outputs = model(**batch, lam=lam)

            optimizer.zero_grad()
            outputs["loss"].backward()
            ratio = generator_discriminator_grad_ratio(model)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
            optimizer.step()
            lr_scheduler.step()

            sums["mlm"] += outputs["loss_mlm"].item()
            sums["rtd"] += outputs["loss_disc"].item()
            sums["rep"] += outputs["replacement_rate"].item()
            sums["ratio"] += ratio
            n_batches += 1
            global_step += 1
            progress.set_postfix(
                lam=f"{lam:.1f}",
                loss=f"{outputs['loss'].item():.4f}",
                mlm=f"{outputs['loss_mlm'].item():.4f}",
                rtd=f"{outputs['loss_disc'].item():.4f}",
            )

        metrics = evaluate_rtd(model, dev_loader, run.device, config.lambda_max, run.max_eval_batches)
        n = max(n_batches, 1)
        row = {
            "epoch": epoch,
            "avg_mlm_loss": round(sums["mlm"] / n, 6),
            "avg_rtd_loss": round(sums["rtd"] / n, 6),
            **{k: round(metrics[k], 6) for k in REPORT_KEYS[2:-1]},
            "grad_norm_ratio": round(sums["ratio"] / n, 6),
        }
        report.append(row)
        logger.info(
            "epoch %d | train MLM %.4f RTD %.4f | dev RTD acc %.4f F1 %.4f rep %.3f",
            epoch, row["avg_mlm_loss"], row["avg_rtd_loss"], row["accuracy"], row["f1"], row["replacement_rate"],
        )

        is_best = row["f1"] > best_f1
        best_f1 = max(best_f1, row["f1"])
        save_epoch(
            paths,
            model,
            epoch,
            row,
            is_best,
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_f1": best_f1,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "report": report,
            },
        )
        save_json(paths.report, report)

    if report:
        plot_report(report, paths.graphs)
    logger.info("Done. Best dev RTD F1 = %.4f (%s)", best_f1, paths.report)
    return report
