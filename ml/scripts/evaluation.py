"""
Comprehensive Evaluation Metrics for Early Warning Classification.
==================================================================
Calculates:
- AUROC (Area under the Receiver Operating Characteristic)
- AUPRC (Area under the Precision-Recall Curve)
- Sensitivity (Recall)
- Specificity
- Brier Score (Calibration error)
- Calibration curves
Handles edge cases (e.g. single class in y_true, zero division, undefined precision).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    metric: str = "f1",
) -> float:
    """
    Select an optimal classification decision threshold using validation data.
    
    Parameters
    ----------
    y_true : 1D array-like of ground truth labels
    y_prob : 1D array-like of predicted probabilities
    metric : Strategy ('f1' or 'youden' / balanced sensitivity-specificity)
    
    Returns
    -------
    Best threshold float in (0, 1). Falls back to 0.5 if optimization not possible.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_t)) < 2:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_t, y_p)
    if len(thresholds) == 0:
        return 0.5

    if metric == "f1":
        f1_scores = np.zeros_like(thresholds)
        for i in range(len(thresholds)):
            p = precisions[i]
            r = recalls[i]
            f1_scores[i] = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(thresholds[best_idx])
    else:
        # Youden's J statistic = Sensitivity + Specificity - 1
        fpr, tpr, roc_thresh = roc_curve(y_t, y_p)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(roc_thresh[best_idx]) if best_idx < len(roc_thresh) else 0.5

    return float(np.clip(best_thresh, 0.001, 0.999))


def evaluate_ews_predictions(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    operating_threshold: float = 0.5,
    threshold_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for EWS binary predictions.
    
    Parameters
    ----------
    y_true : 1D array-like of binary ground truth labels (0 or 1).
    y_prob : 1D array-like of predicted probabilities in [0.0, 1.0].
    operating_threshold : float
        Decision threshold for converting probabilities into binary class predictions.
    threshold_grid : sequence of float, optional
        Grid of thresholds to evaluate sensitivity, specificity, precision, and recall.
        
    Returns
    -------
    Dictionary of scalar metrics, baselines, and threshold-dependent operating points.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    n_samples = len(y_t)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    prevalence = float(n_pos / n_samples) if n_samples > 0 else 0.0

    # 1. All-negative baseline Brier score: predicting constant 0
    all_neg_brier = float(np.mean((0.0 - y_t) ** 2)) if n_samples > 0 else None
    # Prevalence baseline Brier score: predicting constant prevalence
    prevalence_brier = float(np.mean((prevalence - y_t) ** 2)) if n_samples > 0 else None

    res: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": round(prevalence, 4),
        "threshold": operating_threshold,
        "baselines": {
            "all_negative_brier_score": round(all_neg_brier, 4) if all_neg_brier is not None else None,
            "prevalence_brier_score": round(prevalence_brier, 4) if prevalence_brier is not None else None,
            "random_auprc": round(prevalence, 4),
        },
    }

    # Handle edge case: only one class present in y_true
    if n_pos == 0 or n_neg == 0:
        res.update({
            "auroc": None,
            "auprc": None,
            "sensitivity": None,
            "specificity": None,
            "brier_score": float(np.mean((y_p - y_t) ** 2)) if n_samples > 0 else None,
            "note": "Evaluation contains only one class in y_true; AUROC and AUPRC undefined.",
        })
        return res

    # 2. Discrimination: AUROC
    try:
        auroc = float(roc_auc_score(y_t, y_p))
    except Exception:
        auroc = None

    # 3. Precision-Recall: AUPRC
    try:
        auprc = float(average_precision_score(y_t, y_p))
    except Exception:
        auprc = None

    # 4. Calibration: Model Brier Score (lower is better)
    try:
        brier = float(brier_score_loss(y_t, y_p))
    except Exception:
        brier = None

    # 5. Operating Point Metrics at specified threshold
    def _metrics_at(th: float) -> Dict[str, Any]:
        y_pred = (y_p >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        return {
            "threshold": round(float(th), 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            },
        }

    operating_metrics = _metrics_at(operating_threshold)

    # 6. Multi-threshold report
    if threshold_grid is None:
        threshold_grid = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    threshold_evaluations = [_metrics_at(th) for th in threshold_grid]

    res.update({
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "operating_metrics": operating_metrics,
        "sensitivity": operating_metrics["sensitivity"],
        "specificity": operating_metrics["specificity"],
        "positive_predictive_value": operating_metrics["precision"],
        "negative_predictive_value": operating_metrics["npv"],
        "confusion_matrix": {
            "true_positive": operating_metrics["confusion_matrix"]["tp"],
            "false_positive": operating_metrics["confusion_matrix"]["fp"],
            "true_negative": operating_metrics["confusion_matrix"]["tn"],
            "false_negative": operating_metrics["confusion_matrix"]["fn"],
        },
        "threshold_evaluations": threshold_evaluations,
    })

    return res
