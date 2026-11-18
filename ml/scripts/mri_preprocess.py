"""
Shared MRI image preprocessing utilities.

Used by both train_mri.py and the backend inference service.
All preprocessing is kept here so training and inference use identical transforms.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# ViT standard ImageNet normalisation (mean / std per channel)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Ordered class names matching the Brain Tumor MRI Dataset folder structure.
# These four classes are the ONLY labels used by the MRI models.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)


def open_image(source) -> Image.Image:
    """
    Open an image from a file path, bytes, or BytesIO object.
    Raises ValueError on unrecognised format.
    """
    try:
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, io.IOBase):
            img = Image.open(source)
        else:
            img = Image.open(io.BytesIO(source))
        img.load()  # force decode here so errors surface early
        return img.convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc


def preprocess_to_numpy(source) -> np.ndarray:
    """
    Load an image and return a normalised float32 numpy array of shape (3, H, W).
    Compatible with ViT feature extractor expectations.
    """
    img = open_image(source)
    img = img.resize(IMAGE_SIZE, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                # normalise
    arr = np.transpose(arr, (2, 0, 1))                        # (3, H, W)
    return arr


def preprocess_batch(paths: list[Path]) -> np.ndarray:
    """
    Preprocess a list of image paths and return shape (N, 3, H, W).
    Skips files that cannot be opened and logs a warning.
    """
    arrays = []
    failed = []
    for p in paths:
        try:
            arrays.append(preprocess_to_numpy(p))
        except Exception:
            failed.append(str(p))
    if failed:
        import warnings
        warnings.warn(f"Skipped {len(failed)} images that could not be opened.")
    if not arrays:
        raise RuntimeError("No valid images could be loaded from the provided paths.")
    return np.stack(arrays, axis=0)
