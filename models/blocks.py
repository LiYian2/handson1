from __future__ import annotations

import torch
import torch_npu
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def make_activation(name: str = "relu") -> nn.Module:
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}")
    return ACTIVATIONS[key]()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act = make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.use_skip = use_skip

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip:
            out = out + identity

        out = self.act(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        bias = not use_batchnorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if use_batchnorm else nn.Identity()
        self.act = make_activation(activation)
        self.use_skip = use_skip

        final_channels = out_channels * self.expansion
        if stride != 1 or in_channels != final_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, final_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(final_channels) if use_batchnorm else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_skip:
            out = out + identity

        out = self.act(out)
        return out
