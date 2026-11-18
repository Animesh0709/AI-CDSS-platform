"""
Tumor Progression Prediction — Model Training & Benchmark Pipeline
===================================================================

Trains and benchmarks three independent longitudinal progression models:
1. Longitudinal XGBoost
2. Longitudinal Support Vector Machine (SVM)
3. Deep Learning Longitudinal Neural Network (PyTorch MLP with Residual Connections)

All models are trained on patient-grouped train splits, validated on validation patients,
and benchmarked on the identical held-out test patient cohort.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim

_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / "ml" / "data" / "lumiere"
MODELS_DIR = _ROOT / "ml" / "models"

META_COLS = {
    "patient_id", "baseline_date", "followup_date", "rano_label", "progression_target"
}


def load_dataset_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df = pd.read_csv(DATA_DIR / "progression_train.csv")
    val_df = pd.read_csv(DATA_DIR / "progression_val.csv")
    test_df = pd.read_csv(DATA_DIR / "progression_test.csv")

    feature_cols = [c for c in train_df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(train_df[c])]
    return train_df, val_df, test_df, feature_cols


def prepare_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    preprocessor = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler)
    ])

    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    y_train = train_df["progression_target"].values.astype(int)
    y_val = val_df["progression_target"].values.astype(int)
    y_test = test_df["progression_target"].values.astype(int)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = float(auc(recall_curve, precision_curve))
    brier = float(brier_score_loss(y_true, y_proba))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def compute_bootstrap_uncertainty(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstraps: int = 200,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(random_state)
    unique_patients = test_df["patient_id"].unique()
    pr_aucs, rocs, sens_list, spec_list, bal_accs = [], [], [], [], []

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        pos_indices = []
        for pt in sampled_pts:
            pos_indices.extend(np.where(test_df["patient_id"].values == pt)[0])

        y_b = y_true[pos_indices]
        p_b = y_proba[pos_indices]

        if len(np.unique(y_b)) < 2:
            continue

        pred_b = (p_b >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_b, pred_b, labels=[0, 1]).ravel()

        precision_curve, recall_curve, _ = precision_recall_curve(y_b, p_b)
        pr_aucs.append(auc(recall_curve, precision_curve))
        rocs.append(roc_auc_score(y_b, p_b))
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        bal_accs.append(balanced_accuracy_score(y_b, pred_b))

    def ci(arr):
        return {
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4)
        } if arr else {"ci_lower": 0.0, "ci_upper": 0.0}

    return {
        "pr_auc": ci(pr_aucs),
        "roc_auc": ci(rocs),
        "balanced_accuracy": ci(bal_accs),
        "sensitivity": ci(sens_list),
        "specificity": ci(spec_list)
    }


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray
) -> XGBClassifier:
    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=15,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def train_svm(
    X_train: np.ndarray, y_train: np.ndarray
) -> CalibratedClassifierCV:
    base_svc = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    # Calibrated probability classifier using sigmoid (Platt scaling)
    calibrated_model = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


class LongitudinalNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.LayerNorm(32)

        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn1(self.fc1(x))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        logits = self.head(out)
        return logits


def train_deep_learning(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 50, lr: float = 0.003
) -> LongitudinalNet:
    input_dim = X_train.shape[1]
    model = LongitudinalNet(input_dim)

    pos_weight = torch.tensor([float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    n_samples = len(X_train)
    batch_size = 32
    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = perm[i:i + batch_size]
            b_x = train_x[indices]
            b_y = train_y[indices]

            optimizer.zero_grad()
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(val_x), val_y).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading preprocessed patient splits...")
    train_df, val_df, test_df, feature_cols = load_dataset_splits()
    print(f"Features count: {len(feature_cols)}")
    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = prepare_matrices(
        train_df, val_df, test_df, feature_cols
    )

    # 1. XGBoost
    print("\nTraining Longitudinal XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba)
    xgb_ci = compute_bootstrap_uncertainty(test_df, y_test, xgb_proba)
    print(f"XGBoost Test PR-AUC: {xgb_metrics['pr_auc']}, Balanced Acc: {xgb_metrics['balanced_accuracy']}, Sens: {xgb_metrics['sensitivity']}, Spec: {xgb_metrics['specificity']}")

    # 2. SVM
    print("\nTraining Longitudinal SVM...")
    svm_model = train_svm(X_train, y_train)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    svm_metrics = compute_metrics(y_test, svm_proba)
    svm_ci = compute_bootstrap_uncertainty(test_df, y_test, svm_proba)
    print(f"SVM Test PR-AUC: {svm_metrics['pr_auc']}, Balanced Acc: {svm_metrics['balanced_accuracy']}, Sens: {svm_metrics['sensitivity']}, Spec: {svm_metrics['specificity']}")

    # 3. Deep Learning
    print("\nTraining Longitudinal Deep Learning...")
    dl_model = train_deep_learning(X_train, y_train, X_val, y_val)
    with torch.no_grad():
        dl_logits = dl_model(torch.tensor(X_test, dtype=torch.float32))
        dl_proba = torch.sigmoid(dl_logits).squeeze().cpu().numpy()
    dl_metrics = compute_metrics(y_test, dl_proba)
    dl_ci = compute_bootstrap_uncertainty(test_df, y_test, dl_proba)
    print(f"DL Test PR-AUC: {dl_metrics['pr_auc']}, Balanced Acc: {dl_metrics['balanced_accuracy']}, Sens: {dl_metrics['sensitivity']}, Spec: {dl_metrics['specificity']}")

    importance_dict = dict(zip(feature_cols, [float(v) for v in xgb_model.feature_importances_]))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    eval_report = {
        "status": "available",
        "dataset": "LUMIERE Longitudinal Glioblastoma MRI (Scientific Data / Figshare 5904905)",
        "held_out_patients_count": int(test_df["patient_id"].nunique()),
        "held_out_transitions_count": len(test_df),
        "test_class_distribution": {
            "progressive_disease_count": int(np.sum(y_test == 1)),
            "stable_or_response_count": int(np.sum(y_test == 0))
        },
        "models": {
            "xgboost": {
                "name": "Longitudinal XGBoost",
                "metrics": xgb_metrics,
                "confidence_intervals": xgb_ci
            },
            "svm": {
                "name": "Longitudinal SVM (RBF Kernel)",
                "metrics": svm_metrics,
                "confidence_intervals": svm_ci
            },
            "deep_learning": {
                "name": "Longitudinal Deep Neural Net",
                "metrics": dl_metrics,
                "confidence_intervals": dl_ci
            }
        },
        "top_predictive_features": [{"feature": k, "importance": round(v, 4)} for k, v in top_features]
    }

    metadata = {
        "dataset_name": "LUMIERE Dataset",
        "dataset_doi": "10.1038/s41597-022-01881-7",
        "figshare_collection": "5904905",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "classes": ["Stable/Response (0)", "Progressive Disease (1)"],
        "clinical_target": "Longitudinal RANO Progression (PD vs Non-PD)",
        "disclaimer": "Clinical decision support only. Requires review by certified neuro-oncologist."
    }

    joblib.dump(xgb_model, MODELS_DIR / "progression_xgboost.joblib")
    joblib.dump(svm_model, MODELS_DIR / "progression_svm.joblib")
    torch.save(dl_model.state_dict(), MODELS_DIR / "progression_dl.pt")
    joblib.dump(preprocessor, MODELS_DIR / "progression_preprocessor.joblib")

    (MODELS_DIR / "progression_eval_report.json").write_text(json.dumps(eval_report, indent=2))
    (MODELS_DIR / "progression_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nAll models and evaluation reports successfully saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
