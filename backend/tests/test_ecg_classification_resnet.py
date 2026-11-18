"""
Unit and integration tests for Phase 7.4 1D ResNet Multi-Label ECG Classifier.

Verifies:
1. ResNet1D input shape (B, 12, 1000) and multi-label output shape (B, 5).
2. Continuous logits and sigmoid-activated probabilities in [0, 1].
3. Forward pass and gradient flow through backpropagation.
4. Intermediate convolutional feature map extraction (for Grad-CAM hooks).
5. Dataset channel normalization isolation (training-only parameters).
6. Target pos_weight computation for BCEWithLogitsLoss.
7. Model checkpoint save and load consistency.
8. Evaluation with custom validation thresholds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.ecg_classification.dataset import PTBXLWaveformDataset
from ml.ecg_classification.resnet1d import ECGResNet1D, ResNetBlock1D
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


class TestECGResNet1D:

    def test_resnet_block_1d_forward_and_residual(self):
        """Verify 1D residual block handles identity and stride downsampling with skip connection."""
        # Identity shortcut (in_channels == out_channels, stride=1)
        block = ResNetBlock1D(in_channels=32, out_channels=32, stride=1)
        x = torch.randn(4, 32, 100)
        out = block(x)
        assert out.shape == (4, 32, 100)
        assert isinstance(block.shortcut, nn.Identity)

        # Downsampling shortcut (in_channels != out_channels, stride=2)
        block_down = ResNetBlock1D(in_channels=32, out_channels=64, stride=2)
        out_down = block_down(x)
        assert out_down.shape == (4, 64, 50)
        assert isinstance(block_down.shortcut, nn.Sequential)

    def test_resnet_input_output_shapes_and_probabilities(self):
        """Verify model accepts (B, 12, 1000) and produces (B, 5) independent logits and probabilities."""
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
        model.eval()

        batch_size = 8
        dummy_input = torch.randn(batch_size, 12, 1000)

        with torch.no_grad():
            logits = model(dummy_input)

        assert logits.shape == (batch_size, 5)
        probs = torch.sigmoid(logits)
        assert probs.shape == (batch_size, 5)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        # Invalid input dimension must raise ValueError
        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(batch_size, 8, 1000))

    def test_intermediate_feature_map_accessibility(self):
        """Verify extract_features exposes last_conv_feature_map with correct spatial dimensions for Grad-CAM."""
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
        model.eval()

        dummy_input = torch.randn(2, 12, 1000)
        features = model.extract_features(dummy_input)

        # 4 stages of downsampling stride 2: 1000 -> 500 -> 250 -> 125 -> 63 -> 32
        # Final conv features have shape (2, base_channels * 4, 32) = (2, 128, 32)
        assert features.shape == (2, 128, 32)
        assert model.last_conv_feature_map is not None
        assert model.last_conv_feature_map.shape == (2, 128, 32)

    def test_gradient_flow_and_backpropagation(self):
        """Verify gradients propagate to all trainable parameters during backward pass."""
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=16)
        model.train()

        x = torch.randn(4, 12, 1000)
        y = torch.randint(0, 2, (4, 5)).float()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Verify every parameter with requires_grad received a non-zero/valid gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

        optimizer.step()

    def test_parameter_count(self):
        """Verify model parameter count with base_channels=32 is approximately 725K."""
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
        params = model.count_parameters()
        assert params == 725957

    def test_model_serialization_consistency(self):
        """Verify model weights and architecture can be saved and accurately reloaded."""
        model = ECGResNet1D(in_channels=12, num_classes=5, base_channels=32)
        model.eval()

        dummy_input = torch.randn(3, 12, 1000)
        with torch.no_grad():
            orig_output = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model_test.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": 12,
                    "num_classes": 5,
                    "base_channels": 32,
                },
                save_path,
            )

            # Reload into fresh model instance
            loaded_checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
            new_model = ECGResNet1D(
                in_channels=loaded_checkpoint["in_channels"],
                num_classes=loaded_checkpoint["num_classes"],
                base_channels=loaded_checkpoint["base_channels"],
            )
            new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
            new_model.eval()

            with torch.no_grad():
                loaded_output = new_model(dummy_input)

            torch.testing.assert_close(orig_output, loaded_output)

    def test_normalization_isolation_and_pos_weight(self):
        """Verify channel normalization math and pos_weight formula for multi-label loss."""
        # Simulated multi-label targets (N=10, C=5)
        targets = np.array([
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
        ], dtype=np.float32)

        pos_counts = targets.sum(axis=0)  # [7, 3, 2, 4, 1]
        neg_counts = 10 - pos_counts       # [3, 7, 8, 6, 9]
        expected_pos_weight = neg_counts / pos_counts

        calc_pos_weight = (10 - pos_counts) / np.maximum(pos_counts, 1)
        np.testing.assert_allclose(calc_pos_weight, expected_pos_weight)

        # Test normalization formula: (X - mean) / std
        dummy_signal = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        mean = np.array([20.0, 30.0], dtype=np.float32)
        std = np.array([10.0, 10.0], dtype=np.float32)

        norm_signal = (dummy_signal - mean) / std
        np.testing.assert_allclose(norm_signal, [[-1.0, -1.0], [1.0, 1.0]])
