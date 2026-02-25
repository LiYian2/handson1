from __future__ import annotations

from typing import Any, Dict

from models.plainnet import PlainNet
from models.resnet import resnet9, resnet18, resnet34, resnet50
from models.resnet import resnet9_narrow, resnet18_narrow, resnet34_narrow, resnet50_narrow
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
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
    narrow = bool(model_cfg.get("narrow", False))
    kwargs = {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "use_batchnorm": use_batchnorm,
        "activation": activation,
        "dropout": dropout,
        "use_skip": bool(model_cfg.get("use_skip", True)),
    }

    if name == "resnet9":
        if narrow:
            return resnet9_narrow(**kwargs)
        return resnet9(**kwargs)
    if name == "resnet18":
        if narrow:
            return resnet18_narrow(**kwargs)
        return resnet18(**kwargs)
    if name == "resnet34":
        if narrow:
            return resnet34_narrow(**kwargs)
        return resnet34(**kwargs)
    if name == "resnet50":
        if narrow:
            return resnet50_narrow(**kwargs)
        return resnet50(**kwargs)
    if name == "plainnet":
        kwargs.pop("dropout", None)
        return PlainNet(**kwargs)
    
    if name == "resnet20":
        return resnet20(**kwargs)
    if name == "resnet32":
        return resnet32(**kwargs)
    if name == "resnet44":
        return resnet44(**kwargs)
    if name == "resnet56":
        return resnet56(**kwargs)
    if name == "resnet110":
        return resnet110(**kwargs)

    if name == "vgg11":
        kwargs.pop("use_skip", None)
        return VGG11(**kwargs)

    raise ValueError(f"Unknown model: {name}")
