from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from data.dataset import build_dataloaders
from experiments.common import build_device, build_optimizer, build_scheduler, load_config, set_seed
from models import build_model
from utils.trainer import Trainer


def run(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))

    device = build_device(cfg)
    model = build_model(cfg)

    if dry_run:
        x = torch.randn(4, int(cfg["data"].get("input_channels", 1)), int(cfg["data"].get("image_size", 28)), int(cfg["data"].get("image_size", 28)))
        y = model(x)
        print(f"[DRY-RUN] model={cfg['model']['name']} output_shape={tuple(y.shape)} device={device}")
        return

    train_loader, val_loader = build_dataloaders(cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    exp_name = cfg.get("experiment", {}).get("name", "default_experiment")
    output_dir = os.path.join(cfg.get("runtime", {}).get("output_dir", "./outputs"), "results", exp_name)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        save_interval=int(cfg.get("logging", {}).get("save_interval", 5)),
    )

    history = trainer.train(num_epochs=int(cfg["training"].get("num_epochs", 5)))
    from utils.visualization import plot_curves

    plot_curves(history.train_loss, history.val_loss, "loss", os.path.join(output_dir, "loss_curve.png"))
    plot_curves(history.train_acc, history.val_acc, "accuracy", os.path.join(output_dir, "acc_curve.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(args.config, dry_run=args.dry_run)
