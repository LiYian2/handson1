from __future__ import annotations

import torch
import torch.nn as nn

from models.resnet import resnet9


class PlainNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, use_batchnorm: bool = True, activation: str = "relu") -> None:
        super().__init__()
        self.backbone = resnet9(
            in_channels=in_channels,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
            activation=activation,
            use_skip=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
