from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from utils.evaluator import evaluate
from utils.logger import CSVLogger


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        output_dir: str,
        scheduler: _LRScheduler | None = None,
        save_interval: int = 5,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.save_interval = save_interval

        os.makedirs(output_dir, exist_ok=True)
        self.logger = CSVLogger(output_dir)

    def train(self, num_epochs: int) -> TrainHistory:
        history = TrainHistory()
        self.model.to(self.device)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for images, labels, _ in bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = labels.size(0)
                epoch_loss += loss.item() * bs
                epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
                epoch_total += bs

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss = epoch_loss / max(epoch_total, 1)
            train_acc = epoch_correct / max(epoch_total, 1)
            val_metrics = evaluate(self.model, self.val_loader, self.criterion, self.device)

            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_metrics["loss"])
            history.val_acc.append(val_metrics["accuracy"])

            self.logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if epoch % self.save_interval == 0 or epoch == num_epochs:
                ckpt_path = os.path.join(self.output_dir, f"epoch_{epoch}.pt")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                    },
                    ckpt_path,
                )

        return history
