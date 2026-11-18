"""
Phase 8.0: 1D Grad-CAM for Multilabel 12-Lead ECG Abnormality Classification.

Extracts class-specific temporal activation maps from the final convolutional stage of
ECGResNet1D (shape: (B, 128, 32)) and upsamples them to the original 1,000-sample time axis.

Methodological constraints:
1. Target is strictly the raw class logit, not sigmoid probabilities.
2. Supports batch or single-record inputs on CPU and Apple Silicon MPS.
3. Preserves input tensor immutability.
4. Channel-wise gradient averaging over the temporal dimension.
5. ReLU rectification followed by min-max normalization per record.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


class ECGGradCAM1D:
    """
    1D Grad-CAM implementation for 12-lead ECG time-series neural networks.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Trained ECGResNet1D model.
            target_layer: Target convolutional module (defaults to model.layer4).
            target_classes: Ordered superclass names (defaults to DIAGNOSTIC_SUPERCLASSES).
        """
        self.model = model
        self.target_layer = target_layer if target_layer is not None else getattr(model, "layer4", None)
        if self.target_layer is None:
            raise ValueError("Target layer could not be determined. Please supply target_layer explicitly.")

        self.target_classes = target_classes or DIAGNOSTIC_SUPERCLASSES
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.target_classes)}

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target convolutional layer."""
        self._remove_hooks()
        self._activations = None
        self._gradients = None

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._activations = output

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: torch.Tensor) -> None:
            # grad_output is a tuple; first element corresponds to the layer's output gradient
            self._gradients = grad_output[0]

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h_fwd, h_bwd])

    def _remove_hooks(self) -> None:
        """Remove active forward and backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate_cam(
        self,
        inputs: torch.Tensor,
        class_target: Union[str, int],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Generate 1D Grad-CAM temporal attribution map for the specified class.

        Args:
            inputs: Tensor of shape (12, 1000) or (B, 12, 1000).
            class_target: Target superclass name (e.g. 'MI') or integer index in [0, 4].
            normalize: Whether to min-max scale the attribution to [0, 1] per record.

        Returns:
            cam: Temporal attribution tensor of shape (B, 1000) or (1000,) if unbatched.
        """
        was_unbatched = False
        if inputs.ndim == 2:
            if inputs.shape[0] != 12:
                raise ValueError(f"Expected unbatched input shape (12, samples), got {inputs.shape}")
            inputs = inputs.unsqueeze(0)
            was_unbatched = True
        elif inputs.ndim == 3:
            if inputs.shape[1] != 12:
                raise ValueError(f"Expected batched input shape (B, 12, samples), got {inputs.shape}")
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {inputs.ndim}D tensor")

        target_idx: int
        if isinstance(class_target, str):
            if class_target not in self.class_to_idx:
                raise ValueError(f"Unknown class '{class_target}'. Valid classes: {self.target_classes}")
            target_idx = self.class_to_idx[class_target]
        else:
            target_idx = int(class_target)
            if not (0 <= target_idx < len(self.target_classes)):
                raise ValueError(f"Class index {target_idx} out of range [0, {len(self.target_classes)-1}]")

        # Preserve input immutability: clone inputs and retain computational graph
        x = inputs.clone().detach().requires_grad_(True)
        device = x.device
        orig_length = x.shape[-1]

        # Put model in eval mode for inference
        was_training = self.model.training
        self.model.eval()

        self._register_hooks()
        try:
            self.model.zero_grad(set_to_none=True)
            # Forward pass: obtain raw class logits
            logits = self.model(x)
            if logits.ndim != 2 or logits.shape[1] <= target_idx:
                raise RuntimeError(f"Unexpected logits shape {logits.shape} for target index {target_idx}")

            # Target score: sum of raw logits for the chosen class across batch
            target_logits = logits[:, target_idx]
            target_logits.sum().backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

            activations = self._activations.detach()  # Shape: (B, C, L_conv), e.g., (B, 128, 32)
            gradients = self._gradients.detach()      # Shape: (B, C, L_conv), e.g., (B, 128, 32)

            # Global average pooling of gradients along the temporal dimension L_conv
            # weights shape: (B, C, 1)
            weights = gradients.mean(dim=-1, keepdim=True)

            # Check for zero-gradient case (constant feature map or zero-gradient graph)
            if torch.all(weights == 0.0) or torch.all(gradients == 0.0):
                # Return zero attribution map rather than misleading constant or random values
                cam = torch.zeros((x.shape[0], orig_length), dtype=torch.float32, device=device)
                return cam.squeeze(0) if was_unbatched else cam

            # Linear combination of feature maps weighted by pooled gradients
            # cam shape: (B, L_conv)
            cam = (weights * activations).sum(dim=1)

            # Apply ReLU to focus on positive contributions to the selected class
            cam = F.relu(cam)

            # Upsample linearly along time axis to match original temporal resolution (e.g. 32 -> 1,000)
            cam_upsampled = F.interpolate(
                cam.unsqueeze(1),
                size=orig_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # Shape: (B, orig_length)

            # Per-record min-max normalization to [0, 1]
            if normalize:
                b_min = cam_upsampled.amin(dim=-1, keepdim=True)
                b_max = cam_upsampled.amax(dim=-1, keepdim=True)
                denom = b_max - b_min
                denom[denom < 1e-8] = 1.0
                cam_normalized = (cam_upsampled - b_min) / denom
                # If peak was 0, ensure map remains exact zero
                cam_normalized = torch.where(denom <= 1e-8, torch.zeros_like(cam_normalized), cam_normalized)
                res = cam_normalized
            else:
                res = cam_upsampled

            return res.squeeze(0) if was_unbatched else res

        finally:
            self._remove_hooks()
            if was_training:
                self.model.train()
