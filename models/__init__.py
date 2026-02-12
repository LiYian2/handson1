from __future__ import annotations

from typing import Any, Dict

from models.plainnet import PlainNet
from models.resnet import resnet9, resnet18, resnet50
from models.vgg import VGG11


def build_model(cfg: Dict[str, Any]):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    name = model_cfg.get("name", "resnet9").lower()
    in_channels = int(data_cfg.get("input_channels", 1))
    num_classes = int(data_cfg.get("num_classes", 10))
    use_batchnorm = bool(model_cfg.get("use_batchnorm", True))
    activation = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))

    kwargs = {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "use_batchnorm": use_batchnorm,
        "activation": activation,
        "dropout": dropout,
    }

    if name == "resnet9":
        return resnet9(**kwargs)
    if name == "resnet18":
        return resnet18(**kwargs)
    if name == "resnet50":
        return resnet50(**kwargs)
    if name == "plainnet":
        kwargs.pop("dropout", None)
        return PlainNet(**kwargs)
    if name == "vgg11":
        return VGG11(**kwargs)

    raise ValueError(f"Unknown model: {name}")
