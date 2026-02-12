from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import make_activation


class VGG11(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        activation: str = "relu",
        use_batchnorm: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        act = make_activation(activation)
        # One less pooling stage than ImageNet VGG11 to fit 28x28 inputs.
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]

        layers = []
        c = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(c, v, kernel_size=3, padding=1, bias=not use_batchnorm))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(act.__class__())
                c = v

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            act.__class__(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
