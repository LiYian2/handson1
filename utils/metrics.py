from __future__ import annotations

from typing import Dict

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(total)


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def summarize_metrics(loss: float, acc: float) -> Dict[str, float]:
    return {"loss": float(loss), "accuracy": float(acc)}
