from __future__ import annotations

from typing import List, Type

import torch
#import torch_npu
import torch.nn as nn

from models.blocks import BasicBlock, BottleneckBlock, make_activation


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 10,
        widths: List[int] | None = None,
        use_batchnorm: bool = True,
        activation: str = "relu",
        use_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = widths or [64, 128, 256, 512]
        self.act = make_activation(activation)

        bias = not use_batchnorm
        stem_out = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(stem_out) if use_batchnorm else nn.Identity(),
            self.act,
        )

        self.inplanes = stem_out
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)

        final_dim = widths[3] * block.expansion
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(final_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride, use_batchnorm, activation, use_skip):
        layers = [
            block(self.inplanes, planes, stride=stride, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class ResNet_cifar(nn.Module):
    """CIFAR-style ResNet: 3 stages (16/32/64), BasicBlock, depth = 6n+2."""
    def __init__(self,
                 block: Type[nn.Module],
                 layers: List[int],
                 in_channels: int = 1,
                 num_classes: int = 10,
                 widths: List[int] | None = None,
                 use_batchnorm: bool = True,
                 activation: str = "relu",
                 use_skip: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        widths = widths or [16,32,64]
        self.act = make_activation(activation)
        bias = not use_batchnorm
        stem_out = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(stem_out) if use_batchnorm else nn.Identity(),
            self.act,
        )
        self.inplanes = stem_out
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        final_dim = widths[2] * block.expansion
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(final_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride, use_batchnorm, activation, use_skip):
        layers = [
            block(self.inplanes, planes, stride=stride, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, use_batchnorm=use_batchnorm, activation=activation, use_skip=use_skip)
            )
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def resnet9(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [1, 1, 1, 1], widths=[32, 64, 128, 256], **kwargs)
def resnet9_narrow(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [1, 1, 1, 1], widths=[16, 32, 64, 128], **kwargs)

def resnet20(**kwargs) -> ResNet_cifar:
    return ResNet_cifar(BasicBlock, [3, 3, 3], widths=[16, 32, 64], **kwargs)

def resnet32(**kwargs) -> ResNet_cifar:
    return ResNet_cifar(BasicBlock, [5, 5, 5], widths=[16, 32, 64], **kwargs)

def resnet44(**kwargs) -> ResNet_cifar:
    return ResNet_cifar(BasicBlock, [7, 7, 7], widths=[16, 32, 64], **kwargs)

def resnet56(**kwargs) -> ResNet_cifar:
    return ResNet_cifar(BasicBlock, [9, 9, 9], widths=[16, 32, 64], **kwargs)

def resnet110(**kwargs) -> ResNet_cifar:
    return ResNet_cifar(BasicBlock, [18, 18, 18], widths=[16, 32, 64], **kwargs)

def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], widths=[64, 128, 256, 512], **kwargs)
def resnet18_narrow(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], widths=[32, 64, 128, 256], **kwargs)

def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], widths=[64, 128, 256, 512], **kwargs)
def resnet34_narrow(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], widths=[32, 64, 128, 256], **kwargs)

def resnet50(**kwargs) -> ResNet:
    return ResNet(BottleneckBlock, [3, 4, 6, 3], widths=[64, 128, 256, 512], **kwargs)
def resnet50_narrow(**kwargs) -> ResNet:
    return ResNet(BottleneckBlock, [3, 4, 6, 3], widths=[32, 64, 128, 256], **kwargs)
