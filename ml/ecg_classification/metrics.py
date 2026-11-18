"""
Evaluation Metrics for Multi-Label ECG Abnormality Classification.

Calculates per-class:
- Precision
- Recall
- F1-score
- AUROC (Area under ROC curve)
- AUPRC (Average Precision / Area under PR curve)
- Support (positive count) and prevalence

Calculates aggregate:
- Macro F1
- Micro F1
- Macro AUROC
- Macro AUPRC
- Subset Accuracy (Exact match ratio)
- Hamming Loss

Handles edge cases (single-class present, 0 positive samples, undefined metrics).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MultiLabelMetricsEvaluator:
    """
    Computes rigorous per-class and aggregate multi-label classification metrics.
    """

    def __init__(self, target_classes: List[str]):
        self.target_classes = target_classes

    def find_optimal_thresholds_on_val(
        self,
        y_val: np.ndarray,
        y_val_probs: np.ndarray,
        threshold_candidates: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Find optimal binary decision threshold per class on validation data
        to maximize F1-score. Never use test set for this.
        """
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.1, 0.9, 17)

        optimal_thresholds: Dict[str, float] = {}

        for idx, cls_name in enumerate(self.target_classes):
            y_true_col = y_val[:, idx]
            y_prob_col = y_val_probs[:, idx]

            pos_count = int(np.sum(y_true_col == 1))
            neg_count = int(np.sum(y_true_col == 0))

            # If class has 0 positives in validation, default to 0.5 (or candidate median)
            if pos_count == 0 or neg_count == 0:
                optimal_thresholds[cls_name] = float(np.median(threshold_candidates))
                continue

            best_thresh = float(threshold_candidates[0])
            best_f1 = -1.0

            for th in threshold_candidates:
                y_pred = (y_prob_col >= th).astype(int)
                score = f1_score(y_true_col, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = round(float(th), 4)

            optimal_thresholds[cls_name] = best_thresh

        return optimal_thresholds

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate binary targets against probability estimates using fixed thresholds.
        """
        num_samples, num_classes = y_true.shape
        if num_classes != len(self.target_classes):
            raise ValueError(f"Shape mismatch: expected {len(self.target_classes)} classes, got {num_classes}")

        if thresholds is None:
            thresholds = {cls: 0.5 for cls in self.target_classes}

        # Apply thresholds to get binary predictions
        y_pred = np.zeros_like(y_true, dtype=int)
        for idx, cls_name in enumerate(self.target_classes):
            th = thresholds.get(cls_name, 0.5)
            y_pred[:, idx] = (y_probs[:, idx] >= th).astype(int)

        per_class_metrics: Dict[str, Dict[str, Any]] = {}
        auroc_list: List[float] = []
        auprc_list: List[float] = []
        f1_list: List[float] = []

        for idx, cls_name in enumerate(self.target_classes):
            y_t = y_true[:, idx]
            y_p = y_pred[:, idx]
            prob = y_probs[:, idx]

            pos_count = int(np.sum(y_t == 1))
            total_count = len(y_t)
            prevalence = float(pos_count / total_count) if total_count > 0 else 0.0

            prec = float(precision_score(y_t, y_p, zero_division=0))
            rec = float(recall_score(y_t, y_p, zero_division=0))
            f1 = float(f1_score(y_t, y_p, zero_division=0))
            f1_list.append(f1)

            # AUROC & AUPRC (require at least 1 pos and 1 neg sample)
            if pos_count > 0 and (total_count - pos_count) > 0:
                try:
                    auc_val = float(roc_auc_score(y_t, prob))
                    auroc_list.append(auc_val)
                except Exception:
                    auc_val = None

                try:
                    auprc_val = float(average_precision_score(y_t, prob))
                    auprc_list.append(auprc_val)
                except Exception:
                    auprc_val = None
            else:
                auc_val = None
                auprc_val = None

            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            tn = int(np.sum((y_t == 0) & (y_p == 0)))

            per_class_metrics[cls_name] = {
                "support": pos_count,
                "total_samples": total_count,
                "predicted_positive_count": int(np.sum(y_p == 1)),
                "prevalence": round(prevalence, 4),
                "threshold": thresholds.get(cls_name, 0.5),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "auroc": round(auc_val, 4) if auc_val is not None else None,
                "auprc": round(auprc_val, 4) if auprc_val is not None else None,
                "confusion_matrix": {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                },
            }

        # Aggregate metrics
        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro_auroc = float(np.mean(auroc_list)) if auroc_list else None
        macro_auprc = float(np.mean(auprc_list)) if auprc_list else None
        h_loss = float(hamming_loss(y_true, y_pred))
        exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

        return {
            "per_class": per_class_metrics,
            "aggregate": {
                "macro_f1": round(macro_f1, 4),
                "micro_f1": round(micro_f1, 4),
                "macro_auroc": round(macro_auroc, 4) if macro_auroc is not None else None,
                "macro_auprc": round(macro_auprc, 4) if macro_auprc is not None else None,
                "hamming_loss": round(h_loss, 4),
                "subset_accuracy": round(exact_match, 4),
            },
        }
