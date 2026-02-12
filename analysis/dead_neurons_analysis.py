from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import build_dataloaders
from experiments.common import build_device, load_config
from models import build_model


def estimate_dead_relu_ratio(model: torch.nn.Module, inputs: torch.Tensor, threshold: float = 1e-8) -> Dict[str, float]:
    ratios = {}

    hooks = []
    activations = {}

    def save_hook(name):
        def fn(_, __, output):
            activations[name] = output.detach()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(save_hook(name)))

    _ = model(inputs)
    for name, act in activations.items():
        dead = (act.abs() <= threshold).float().mean().item()
        ratios[name] = dead

    for h in hooks:
        h.remove()

    return ratios


def main(config: str):
    cfg = load_config(config)
    device = build_device(cfg)
    model = build_model(cfg).to(device).eval()

    train_loader, _ = build_dataloaders(cfg)
    images, _, _ = next(iter(train_loader))
    images = images.to(device)

    ratios = estimate_dead_relu_ratio(model, images)
    for k, v in ratios.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp1_baseline.yaml")
    args = parser.parse_args()
    main(args.config)
