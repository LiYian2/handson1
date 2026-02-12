from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import torch
import yaml


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    merged: Dict[str, Any] = {}
    for parent in cfg.get("extends", []):
        parent_cfg = load_config(parent)
        merged = deep_update(merged, parent_cfg)

    cfg.pop("extends", None)
    return deep_update(merged, cfg)


def build_device(cfg: Dict[str, Any]) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(want)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg.get("type", "SGD")
    lr = float(opt_cfg.get("lr", 0.1))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    sch_cfg = cfg["training"].get("lr_scheduler", {})
    name = sch_cfg.get("type", "CosineAnnealingLR")

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sch_cfg.get("T_max", 50)))
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    return None
