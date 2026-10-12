"""
MRI Brain Tumor Classification — Training Script
=================================================

Trains three models on the Brain Tumor MRI Dataset:
  1. XGBoost   — on ViT-base-patch16-224 image embeddings
  2. SVM (RBF) — on the same ViT embeddings
  3. ViT       — fine-tuned end-to-end as a 4-class classifier (if torch available)

Dataset expected at:
  ml/data/brain_tumor/Training/{glioma,meningioma,notumor,pituitary}/
  ml/data/brain_tumor/Testing/{glioma,meningioma,notumor,pituitary}/

Artifacts saved to ml/models/:
  mri_xgboost.joblib
  mri_svm.joblib
  mri_vit/              (PyTorch model, config, label mapping)
  mri_eval_report.json  (accuracy, precision, recall, F1, confusion matrices)
  mri_metadata.json     (provenance, class mapping, limitations)

Usage:
  cd /path/to/project
  python ml/scripts/train_mri.py [--skip-vit] [--data-dir ml/data/brain_tumor]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Resolve paths relative to this script's location
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ML_DIR       = PROJECT_ROOT / "ml"
MODELS_DIR   = ML_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Add scripts dir to path for mri_preprocess import
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mri_preprocess import CLASS_NAMES, NUM_CLASSES, preprocess_batch  # noqa: E402


# ── Dataset helpers ───────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(split_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Walk a split directory (Training or Testing) and collect image paths + labels.
    Returns (paths, labels) in matched order.
    Raises FileNotFoundError if the directory or expected class folders are missing.
    """
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split directory not found: {split_dir}\n"
            "Place the Brain Tumor MRI Dataset at ml/data/brain_tumor/ with "
            "Training/ and Testing/ subdirectories, each containing four class folders: "
            + ", ".join(CLASS_NAMES)
        )

    paths, labels = [], []
    found_classes = []
    for cls in CLASS_NAMES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            warnings.warn(f"Class folder not found (will be skipped): {cls_dir}")
            continue
        found_classes.append(cls)
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
        paths.extend(files)
        labels.extend([cls] * len(files))
        print(f"  [{cls}] {len(files)} images")

    if not paths:
        raise RuntimeError(f"No image files found under {split_dir}")

    print(f"  Total: {len(paths)} images in {len(found_classes)} classes")
    return paths, labels


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


# ── ViT embedding extraction ──────────────────────────────────────────────────

def extract_vit_embeddings(
    paths: list[Path],
    batch_size: int = 32,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    Extract CLS-token embeddings from google/vit-base-patch16-224.
    Caches results to cache_path (npy) if provided.
    Returns array of shape (N, 768).
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}")
        return np.load(str(cache_path))

    try:
        import torch
        from transformers import ViTModel
        ImageProcessor = _get_image_processor()
    except ImportError:
        raise ImportError(
            "torch and transformers are required for ViT embedding extraction.\n"
            "Install them: pip install torch torchvision transformers"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    print(f"  Loading ViT model: google/vit-base-patch16-224 ...")
    feature_extractor = ImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model.eval()

    from PIL import Image as PILImage
    all_embeddings = []
    total_batches = (len(paths) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i: i + batch_size]
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches or batch_num == 1:
                print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch_paths)} images) ...", flush=True)
            images = []
            for p in batch_paths:
                try:
                    img = PILImage.open(p).convert("RGB")
                    images.append(img)
                except Exception as exc:
                    warnings.warn(f"Skipping {p}: {exc}")
                    images.append(PILImage.new("RGB", (224, 224)))  # placeholder black image

            inputs = feature_extractor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
            all_embeddings.append(cls_tokens)

    print()  # newline after \r
    embeddings = np.vstack(all_embeddings)  # (N, 768)

    if cache_path:
        np.save(str(cache_path), embeddings)
        print(f"  Cached embeddings → {cache_path.name}")

    return embeddings


# ── Evaluation helpers ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """Return a structured evaluation dict with per-class and overall metrics."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_names).tolist()

    per_class = {}
    for cls in class_names:
        r = report.get(cls, {})
        per_class[cls] = {
            "precision": round(r.get("precision", 0.0), 4),
            "recall":    round(r.get("recall",    0.0), 4),
            "f1":        round(r.get("f1-score",  0.0), 4),
            "support":   int(r.get("support",     0)),
        }

    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision":  round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall":     round(float(recall_score(   y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1":         round(float(f1_score(       y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class":        per_class,
        "confusion_matrix": cm,
        "class_names":      class_names,
    }


# ── ViT fine-tuning ───────────────────────────────────────────────────────────

def finetune_vit(
    train_paths: list[Path], train_labels: list[str],
    test_paths:  list[Path], test_labels:  list[str],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
) -> dict:
    """
    Fine-tune google/vit-base-patch16-224 as a 4-class brain tumour classifier.
    Saves model to output_dir. Returns evaluation metrics dict.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import ViTForImageClassification
        ImageProcessor = _get_image_processor()
    except ImportError:
        raise ImportError("torch and transformers are required for ViT fine-tuning.")

    from PIL import Image as PILImage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Fine-tuning ViT on device: {device}")

    label2id = {c: i for i, c in enumerate(CLASS_NAMES)}
    id2label = {i: c for i, c in enumerate(CLASS_NAMES)}

    class MRIDataset(Dataset):
        def __init__(self, paths, labels, feature_extractor):
            self.paths = paths
            self.labels = [label2id[l] for l in labels]
            self.fe = feature_extractor

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = PILImage.open(self.paths[idx]).convert("RGB")
            except Exception:
                img = PILImage.new("RGB", (224, 224))
            inputs = self.fe(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, self.labels[idx]

    feature_extractor = ImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_ds = MRIDataset(train_paths, train_labels, feature_extractor)
    test_ds  = MRIDataset(test_paths,  test_labels,  feature_extractor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (pixel_values, labels_batch) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels_batch = labels_batch.to(device)
            optimiser.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"    Epoch {epoch+1}/{epochs} [step {step+1}/{len(train_loader)}] — batch loss: {loss.item():.4f}", flush=True)
        print(f"  Epoch {epoch+1}/{epochs} completed — avg loss: {total_loss/len(train_loader):.4f}", flush=True)

    # Evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for pixel_values, labels_batch in test_loader:
            pixel_values = pixel_values.to(device)
            logits = model(pixel_values=pixel_values).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels_batch.cpu().numpy() if hasattr(labels_batch, "cpu") else labels_batch.numpy())

    y_pred_str = [id2label[int(p)] for p in all_preds]
    y_true_str = [id2label[int(t)] for t in all_true]

    metrics = compute_metrics(y_true_str, y_pred_str, CLASS_NAMES)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"id2label": id2label, "label2id": label2id}, indent=2)
    )
    print(f"  ViT model saved → {output_dir}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MRI brain tumor classifiers")
    parser.add_argument("--data-dir",  type=Path, default=ML_DIR / "data" / "brain_tumor",
                        help="Root directory containing Training/ and Testing/ subdirs")
    parser.add_argument("--skip-vit",  action="store_true",
                        help="Skip ViT fine-tuning (XGBoost and SVM still trained)")
    parser.add_argument("--vit-epochs", type=int, default=3,
                        help="Number of epochs for ViT fine-tuning (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding extraction (default: 32)")
    args = parser.parse_args()

    try:
        import torch
        torch.set_num_threads(8)
    except Exception:
        pass

    DATA_DIR   = args.data_dir
    TRAIN_DIR  = DATA_DIR / "Training"
    TEST_DIR   = DATA_DIR / "Testing"

    print("=" * 60)
    print("  AI-CDSS MRI Brain Tumor Classification — Training")
    print("=" * 60)

    # ── 1. Verify dataset ─────────────────────────────────────────────────────
    print("\n[1/6] Verifying dataset ...")
    print(f"  Data root: {DATA_DIR}")
    print("  Training split:")
    train_paths, train_labels = collect_images(TRAIN_DIR)
    print("  Testing split:")
    test_paths, test_labels   = collect_images(TEST_DIR)

    label_counts_train = {c: train_labels.count(c) for c in CLASS_NAMES if c in train_labels}
    label_counts_test  = {c: test_labels.count(c)  for c in CLASS_NAMES if c in test_labels}
    print(f"\n  Class mapping (0→{CLASS_NAMES})")
    print(f"  Train counts: {label_counts_train}")
    print(f"  Test  counts: {label_counts_test}")

    # ── 2. Extract ViT embeddings ──────────────────────────────────────────────
    print("\n[2/6] Extracting ViT embeddings ...")
    embed_train_cache = DATA_DIR / "embeddings_train.npy"
    embed_test_cache  = DATA_DIR / "embeddings_test.npy"

    print("  Training set:")
    X_train = extract_vit_embeddings(train_paths, batch_size=args.batch_size, cache_path=embed_train_cache)
    print(f"  Shape: {X_train.shape}")

    print("  Test set:")
    X_test = extract_vit_embeddings(test_paths, batch_size=args.batch_size, cache_path=embed_test_cache)
    print(f"  Shape: {X_test.shape}")

    le = LabelEncoder().fit(CLASS_NAMES)
    y_train_enc = le.transform(train_labels)
    y_test_enc  = le.transform(test_labels)

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    print("\n[3/6] Training XGBoost ...")
    from xgboost import XGBClassifier
    t0 = time.time()
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train_enc)
    xgb_preds_enc = xgb_model.predict(X_test)
    xgb_preds = le.inverse_transform(xgb_preds_enc)
    xgb_metrics = compute_metrics(test_labels, list(xgb_preds), CLASS_NAMES)
    print(f"  XGBoost accuracy: {xgb_metrics['accuracy']:.4f}  macro-F1: {xgb_metrics['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(xgb_model, MODELS_DIR / "mri_xgboost.joblib")
    joblib.dump(le, MODELS_DIR / "mri_label_encoder.joblib")
    print(f"  Saved → mri_xgboost.joblib, mri_label_encoder.joblib")

    # ── 4. Train SVM ──────────────────────────────────────────────────────────
    print("\n[4/6] Training SVM (RBF kernel) ...")
    t0 = time.time()
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
    ])
    svm_model.fit(X_train, train_labels)
    svm_preds = svm_model.predict(X_test)
    svm_metrics = compute_metrics(test_labels, list(svm_preds), CLASS_NAMES)
    print(f"  SVM accuracy: {svm_metrics['accuracy']:.4f}  macro-F1: {svm_metrics['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(svm_model, MODELS_DIR / "mri_svm.joblib")
    print(f"  Saved → mri_svm.joblib")

    # ── 5. Fine-tune ViT ──────────────────────────────────────────────────────
    vit_metrics = None
    vit_dir = MODELS_DIR / "mri_vit"
    if args.skip_vit:
        print("\n[5/6] Skipping ViT fine-tuning (--skip-vit flag set).")
    else:
        print(f"\n[5/6] Fine-tuning ViT ({args.vit_epochs} epochs) ...")
        try:
            vit_metrics = finetune_vit(
                train_paths, train_labels,
                test_paths,  test_labels,
                vit_dir,
                epochs=args.vit_epochs,
                batch_size=16,
            )
            print(f"  ViT accuracy: {vit_metrics['accuracy']:.4f}  macro-F1: {vit_metrics['macro_f1']:.4f}")
        except ImportError as exc:
            print(f"  WARNING: {exc}")
            print("  ViT fine-tuning skipped — install torch + transformers to enable.")

    # ── 6. Save evaluation report and metadata ────────────────────────────────
    print("\n[6/6] Saving evaluation report and metadata ...")
    eval_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name":        "Brain Tumor MRI Dataset",
            "root":        str(DATA_DIR),
            "train_counts": label_counts_train,
            "test_counts":  label_counts_test,
            "class_names":  CLASS_NAMES,
            "class_mapping": {i: c for i, c in enumerate(CLASS_NAMES)},
        },
        "xgboost": xgb_metrics,
        "svm":     svm_metrics,
        "vit":     vit_metrics if vit_metrics else {"status": "not_trained"},
    }
    report_path = MODELS_DIR / "mri_eval_report.json"
    report_path.write_text(json.dumps(eval_report, indent=2))
    print(f"  Evaluation report → {report_path.name}")

    metadata = {
        "status": "trained",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Brain MRI tumour classification (4 classes)",
        "classes": CLASS_NAMES,
        "class_mapping": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "models": {
            "xgboost": {
                "description": "XGBoost classifier on ViT-base-patch16-224 CLS embeddings",
                "artifact": "mri_xgboost.joblib",
                "accuracy": xgb_metrics["accuracy"],
                "macro_f1": xgb_metrics["macro_f1"],
            },
            "svm": {
                "description": "SVM (RBF, C=10) with StandardScaler on ViT embeddings",
                "artifact": "mri_svm.joblib",
                "accuracy": svm_metrics["accuracy"],
                "macro_f1": svm_metrics["macro_f1"],
            },
            "vit": {
                "description": "Fine-tuned google/vit-base-patch16-224 (4-class head, 3 epochs)",
                "artifact": "mri_vit/",
                "accuracy": vit_metrics["accuracy"] if vit_metrics else "unavailable",
                "macro_f1": vit_metrics["macro_f1"] if vit_metrics else "unavailable",
            },
        },
        "preprocessing": {
            "input_size": "224x224 RGB",
            "normalisation": "ImageNet mean/std",
            "embedding_model": "google/vit-base-patch16-224 CLS token (768-dim)",
        },
        "dataset_provenance": (
            "Brain Tumor MRI Dataset — typically sourced from Kaggle "
            "(sartajbhuvaji/brain-tumor-classification-mri). "
            "Verify licence and terms before use. "
            "Training/Testing split is dataset-provided (not randomly split here)."
        ),
        "limitations": [
            "Models trained on a single dataset; generalisation to other MRI scanners / populations is unknown.",
            "Outputs are AI-assisted classification only and require review by a qualified clinician.",
            "ViT embeddings are frozen during XGBoost/SVM training — a jointly fine-tuned pipeline may outperform.",
            "No data augmentation applied during embedding extraction.",
            "Class imbalance, if present in the dataset, is not corrected.",
        ],
        "clinical_safety_notice": (
            "All predictions are labelled 'AI-assisted classification — Predicted class'. "
            "Outputs are NOT a diagnosis. Clinician review is mandatory before any clinical action."
        ),
    }
    (MODELS_DIR / "mri_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Metadata → mri_metadata.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  XGBoost  accuracy={xgb_metrics['accuracy']:.4f}  F1={xgb_metrics['macro_f1']:.4f}")
    print(f"  SVM      accuracy={svm_metrics['accuracy']:.4f}  F1={svm_metrics['macro_f1']:.4f}")
    if vit_metrics:
        print(f"  ViT      accuracy={vit_metrics['accuracy']:.4f}  F1={vit_metrics['macro_f1']:.4f}")
    else:
        print("  ViT      [not trained]")
    print(f"\n  Artifacts saved to: {MODELS_DIR}")
    print(f"  Full report: {report_path}")


if __name__ == "__main__":
    main()
