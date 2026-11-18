"""
MRI Brain Tumor Classification — Inference Service
===================================================

Provides cached artifact loaders and a predict_all() function that runs
an uploaded image through XGBoost, SVM, and (optionally) fine-tuned ViT.

Model artifacts live at ml/models/:
  mri_xgboost.joblib
  mri_label_encoder.joblib
  mri_svm.joblib
  mri_vit/  (optional, requires torch + transformers)

If an artifact is missing the corresponding model returns status="unavailable".
Runtime errors are caught per-model so one failure does not crash the others.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ── Resolve paths ─────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[2]   # backend/
_ROOT        = _BACKEND_DIR.parent                   # project root
_ML_MODELS   = _ROOT / "ml" / "models"
_ML_SCRIPTS  = _ROOT / "ml" / "scripts"

# Add ml/scripts to sys.path so we can import mri_preprocess
if str(_ML_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ML_SCRIPTS))

# ── Class names (must match training) ─────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ── Cached artifact loaders ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_xgboost():
    """Returns (xgb_model, label_encoder) or raises FileNotFoundError."""
    import joblib
    model_path = _ML_MODELS / "mri_xgboost.joblib"
    le_path    = _ML_MODELS / "mri_label_encoder.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost MRI model not found: {model_path}")
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {le_path}")
    return joblib.load(model_path), joblib.load(le_path)


@lru_cache(maxsize=1)
def _load_svm():
    """Returns svm_pipeline or raises FileNotFoundError."""
    import joblib
    model_path = _ML_MODELS / "mri_svm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"SVM MRI model not found: {model_path}")
    return joblib.load(model_path)


def _get_image_processor():
    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor
    except ImportError:
        try:
            from transformers import ViTImageProcessor
            return ViTImageProcessor
        except ImportError:
            from transformers import ViTFeatureExtractor
            return ViTFeatureExtractor


@lru_cache(maxsize=1)
def _load_vit_embedding_model():
    """
    Loads google/vit-base-patch16-224 as an embedding extractor (CLS token).
    Returns (vit_model, feature_extractor, device) or raises ImportError / FileNotFoundError.
    """
    import torch
    from transformers import ViTModel
    ImageProcessor = _get_image_processor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe    = ImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model.eval()
    return model, fe, device


@lru_cache(maxsize=1)
def _load_vit_classifier():
    """
    Loads the fine-tuned ViT classifier from ml/models/mri_vit/.
    Returns (vit_model, feature_extractor, id2label, device).
    Raises FileNotFoundError if not trained yet.
    """
    import torch
    from transformers import ViTForImageClassification
    ImageProcessor = _get_image_processor()
    vit_dir = _ML_MODELS / "mri_vit"
    if not vit_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned ViT model not found at {vit_dir}. "
            "Run ml/scripts/train_mri.py to generate it."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe     = ImageProcessor.from_pretrained(str(vit_dir))
    model  = ViTForImageClassification.from_pretrained(str(vit_dir)).to(device)
    model.eval()
    label_path = vit_dir / "label_mapping.json"
    if label_path.exists():
        mapping = json.loads(label_path.read_text())
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
    return model, fe, id2label, device


# ── ViT embedding extraction (single image) ───────────────────────────────────

def _extract_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Extract a (768,) CLS-token embedding from raw image bytes using the
    base ViT model (not the fine-tuned classifier).
    """
    import torch
    from PIL import Image as PILImage
    import io

    vit_model, fe, device = _load_vit_embedding_model()
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = fe(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)
    return cls_token  # (1, 768)


# ── Per-model inference ───────────────────────────────────────────────────────

def _predict_xgboost(image_bytes: bytes) -> dict[str, Any]:
    try:
        model, le = _load_xgboost()
    except FileNotFoundError as exc:
        return {"model_name": "XGBoost", "status": "unavailable", "message": str(exc)}
    except Exception as exc:
        return {"model_name": "XGBoost", "status": "unavailable", "message": f"Failed to load: {exc}"}

    try:
        embedding = _extract_embedding(image_bytes)  # (1, 768)
        proba = model.predict_proba(embedding)[0]     # (n_classes,)
        pred_idx = int(np.argmax(proba))
        pred_class = le.inverse_transform([pred_idx])[0]
        classes = list(le.classes_)
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        return {
            "model_name": "XGBoost",
            "status": "ok",
            "predicted_class": pred_class,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": prob_dict,
        }
    except Exception as exc:
        return {"model_name": "XGBoost", "status": "error", "message": str(exc)}


def _predict_svm(image_bytes: bytes) -> dict[str, Any]:
    try:
        model = _load_svm()
    except FileNotFoundError as exc:
        return {"model_name": "SVM", "status": "unavailable", "message": str(exc)}
    except Exception as exc:
        return {"model_name": "SVM", "status": "unavailable", "message": f"Failed to load: {exc}"}

    try:
        embedding = _extract_embedding(image_bytes)  # (1, 768)
        proba = model.predict_proba(embedding)[0]
        classes = model.classes_
        pred_idx = int(np.argmax(proba))
        pred_class = classes[pred_idx]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        return {
            "model_name": "SVM",
            "status": "ok",
            "predicted_class": pred_class,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": prob_dict,
        }
    except Exception as exc:
        return {"model_name": "SVM", "status": "error", "message": str(exc)}


def _predict_vit(image_bytes: bytes) -> dict[str, Any]:
    try:
        vit_model, fe, id2label, device = _load_vit_classifier()
    except FileNotFoundError as exc:
        return {"model_name": "ViT (fine-tuned)", "status": "unavailable", "message": str(exc)}
    except ImportError:
        return {
            "model_name": "ViT (fine-tuned)",
            "status": "unavailable",
            "message": "torch and transformers are not installed in the backend environment.",
        }
    except Exception as exc:
        return {"model_name": "ViT (fine-tuned)", "status": "unavailable", "message": f"Failed to load: {exc}"}

    try:
        import io
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = fe(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = vit_model(**inputs).logits  # (1, n_classes)
        proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(np.argmax(proba))
        pred_class = id2label[pred_idx]
        prob_dict = {id2label[i]: round(float(p), 4) for i, p in enumerate(proba)}
        return {
            "model_name": "ViT (fine-tuned)",
            "status": "ok",
            "predicted_class": pred_class,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": prob_dict,
        }
    except Exception as exc:
        return {"model_name": "ViT (fine-tuned)", "status": "error", "message": str(exc)}


# ── Public API ────────────────────────────────────────────────────────────────

def predict_all(image_bytes: bytes) -> list[dict[str, Any]]:
    """
    Run image bytes through all three MRI models.
    Each model's result is independent — one failure does not affect others.
    Returns a list of result dicts matching SingleModelPrediction schema.
    """
    # XGBoost and SVM share the ViT embedding, so we optimise by trying to
    # load the embedding model once; if unavailable both report "unavailable".
    results = [
        _predict_xgboost(image_bytes),
        _predict_svm(image_bytes),
        _predict_vit(image_bytes),
    ]
    return results


def load_eval_report() -> dict[str, Any] | None:
    """Load mri_eval_report.json. Returns None if not found."""
    path = _ML_MODELS / "mri_eval_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_model_metadata() -> dict[str, Any]:
    """Load mri_metadata.json. Returns a minimal unavailable dict if not found."""
    path = _ML_MODELS / "mri_metadata.json"
    if not path.exists():
        return {
            "status": "not_trained",
            "message": (
                "Model metadata not found. "
                "Run ml/scripts/train_mri.py after placing the dataset at ml/data/brain_tumor/."
            ),
        }
    return json.loads(path.read_text())
