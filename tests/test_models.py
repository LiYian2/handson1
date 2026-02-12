import torch

from models import build_model


BASE_CFG = {
    "data": {"input_channels": 1, "num_classes": 10},
    "model": {"name": "resnet9", "use_batchnorm": True, "activation": "relu", "dropout": 0.0},
}


def _forward(name: str):
    cfg = {**BASE_CFG, "model": {**BASE_CFG["model"], "name": name}}
    model = build_model(cfg)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)


def test_resnet9_forward():
    _forward("resnet9")


def test_resnet18_forward():
    _forward("resnet18")


def test_resnet50_forward():
    _forward("resnet50")


def test_plainnet_forward():
    _forward("plainnet")


def test_vgg11_forward():
    _forward("vgg11")
