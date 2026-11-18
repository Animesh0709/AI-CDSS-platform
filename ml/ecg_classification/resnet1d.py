"""
PyTorch 1D ResNet Architecture for Multi-Label 12-Lead ECG Abnormality Classification.

Architecture:
- Input shape: (Batch, 12 channels, 1000 temporal samples) at 100 Hz.
- Initial Convolution + BatchNorm + ReLU.
- 4 Residual Stages with Conv1d, BatchNorm1d, ReLU, and progressive downsampling (stride 2).
- Global Average Pooling over time dimension.
- Fully connected linear layer producing 5 independent class logits.
- Structure provides accessible intermediate activation feature maps for future Grad-CAM hook registration.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    """Residual convolutional block with skip-connection and optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """
    1D ResNet for 12-Lead ECG Multi-Label Abnormality Classification.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Initial feature projection (12 channels -> base_channels)
        self.init_conv = nn.Conv1d(
            in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False
        )
        self.init_bn = nn.BatchNorm1d(base_channels)
        self.init_relu = nn.ReLU(inplace=True)

        # 4 Residual Stages: progressively increase channels and halve temporal resolution
        # Stage 1: 32 channels, length 500 -> 250
        self.layer1 = ResNetBlock1D(base_channels, base_channels * 2, stride=2, kernel_size=kernel_size)
        # Stage 2: 64 channels, length 250 -> 125
        self.layer2 = ResNetBlock1D(base_channels * 2, base_channels * 4, stride=2, kernel_size=kernel_size)
        # Stage 3: 128 channels, length 125 -> 63
        self.layer3 = ResNetBlock1D(base_channels * 4, base_channels * 4, stride=2, kernel_size=kernel_size)
        # Stage 4: 128 channels, length 63 -> 32
        self.layer4 = ResNetBlock1D(base_channels * 4, base_channels * 4, stride=2, kernel_size=kernel_size)

        # Explainability access hook target: the final convolutional feature map
        self.last_conv_feature_map: Optional[torch.Tensor] = None

        # Global average pooling & multi-label classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the final convolutional feature map (for Grad-CAM explainability)."""
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.init_relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        self.last_conv_feature_map = out
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 12, 1000)
        Returns:
            logits: Output tensor of shape (batch_size, 5)
        """
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shape (batch, {self.in_channels}, samples), got {x.shape}")

        features = self.extract_features(x)
        pooled = self.gap(features).squeeze(-1)
        logits = self.fc(pooled)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
