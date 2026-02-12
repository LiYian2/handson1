from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_confusion_matrix(cm: torch.Tensor, save_path: str, title: str = "Confusion Matrix") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm.cpu().numpy(), annot=False, cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_curves(train_values, val_values, name: str, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_values, label=f"train_{name}")
    ax.plot(val_values, label=f"val_{name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
