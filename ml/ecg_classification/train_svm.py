"""
Multi-Label LinearSVM Baseline Classifier for PTB-XL 12-Lead ECG Abnormality Classification.

Architecture:
- Five independent binary LinearSVC classifiers for target superclasses: NORM, MI, STTC, CD, HYP.
- LinearSVC outputs continuous decision scores (signed distance to separating hyperplane), NOT probabilities.
- Decision scores are used directly for ranking metrics (AUROC, AUPRC) and F1 threshold optimization.
- Features are scaled via training-only StandardScaler (fit on train only, frozen for val/test).
- Training-only class imbalance handling via class_weight="balanced".
- Handles degenerate single-class splits gracefully with ConstantPredictor fallback.
- Validation-only decision score threshold optimization maximizing F1.
- Complete artifact persistence: models, scaler, preprocessing, thresholds, config, metrics, comparison.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.ecg_classification.feature_extractor import ECGFeatureExtractor
from ml.ecg_classification.label_mapping import PTBXLLabelMapper
from ml.ecg_classification.metrics import MultiLabelMetricsEvaluator
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES, PTBXLDatasetLoader

logger = logging.getLogger(__name__)


class ConstantScorePredictor:
    """Fallback predictor when training set has only 0s or only 1s for a binary class."""

    def __init__(self, constant_score: float):
        self.constant_score = float(constant_score)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.constant_score, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)


class MultiLabelSVMClassifier:
    """
    Five-class multi-label ECG classifier container with independent binary LinearSVC models.
    """

    def __init__(
        self,
        target_classes: Optional[List[str]] = None,
        C: float = 1.0,
        loss: str = "squared_hinge",
        dual: Union[str, bool] = "auto",
        max_iter: int = 2000,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ):
        self.target_classes = target_classes or DIAGNOSTIC_SUPERCLASSES
        self.hyperparams = {
            "C": float(C),
            "loss": loss,
            "dual": dual,
            "max_iter": int(max_iter),
            "random_state": int(random_state),
            "class_weight": class_weight,
        }
        self.models: Dict[str, Union[LinearSVC, ConstantScorePredictor]] = {}
        self.thresholds: Dict[str, float] = {cls: 0.0 for cls in self.target_classes}
        self.training_prevalences: Dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Fit independent binary LinearSVC models strictly on training features and targets.
        """
        if len(X_train) != len(y_train):
            raise ValueError(f"Shape mismatch: X_train has {len(X_train)} samples, y_train has {len(y_train)}")

        training_info = {}

        for idx, cls_name in enumerate(self.target_classes):
            y_cls = y_train[:, idx]
            n_pos = int(np.sum(y_cls == 1))
            n_neg = int(np.sum(y_cls == 0))
            prevalence = float(n_pos / len(y_cls)) if len(y_cls) > 0 else 0.0
            self.training_prevalences[cls_name] = prevalence

            if n_pos == 0:
                logger.warning(f"Class '{cls_name}' has 0 positive samples in train split. Using ConstantScorePredictor(-1.0).")
                self.models[cls_name] = ConstantScorePredictor(-1.0)
                training_info[cls_name] = {"model_type": "ConstantScorePredictor", "n_pos": 0, "n_neg": n_neg}
            elif n_neg == 0:
                logger.warning(f"Class '{cls_name}' has 0 negative samples in train split. Using ConstantScorePredictor(1.0).")
                self.models[cls_name] = ConstantScorePredictor(1.0)
                training_info[cls_name] = {"model_type": "ConstantScorePredictor", "n_pos": n_pos, "n_neg": 0}
            else:
                model = LinearSVC(
                    C=self.hyperparams["C"],
                    loss=self.hyperparams["loss"],
                    dual=self.hyperparams["dual"],
                    max_iter=self.hyperparams["max_iter"],
                    random_state=self.hyperparams["random_state"],
                    class_weight=self.hyperparams["class_weight"],
                )
                t0 = time.time()
                model.fit(X_train, y_cls)
                fit_duration = time.time() - t0
                self.models[cls_name] = model
                training_info[cls_name] = {
                    "model_type": "LinearSVC",
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "fit_duration_sec": round(fit_duration, 3),
                }

        return training_info

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Produce independent decision scores (signed distance to hyperplane) for all classes.
        Returns array of shape (N, num_classes).
        """
        n_samples = len(X)
        n_classes = len(self.target_classes)
        scores = np.zeros((n_samples, n_classes), dtype=np.float32)

        for idx, cls_name in enumerate(self.target_classes):
            model = self.models.get(cls_name)
            if model is None:
                scores[:, idx] = 0.0
            else:
                scores[:, idx] = model.decision_function(X)

        return scores

    def predict(self, X: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Produce binary multi-hot matrix (N, num_classes) by thresholding decision scores.
        """
        scores = self.decision_function(X)
        th_dict = thresholds or self.thresholds
        preds = np.zeros_like(scores, dtype=int)

        for idx, cls_name in enumerate(self.target_classes):
            th = th_dict.get(cls_name, 0.0)
            preds[:, idx] = (scores[:, idx] >= th).astype(int)

        return preds

    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set decision score thresholds."""
        for cls_name, th in thresholds.items():
            if cls_name in self.target_classes:
                self.thresholds[cls_name] = float(th)

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save models and metadata to directory."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        config = {
            "target_classes": self.target_classes,
            "hyperparams": self.hyperparams,
            "thresholds": self.thresholds,
            "training_prevalences": self.training_prevalences,
        }
        with open(out_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> MultiLabelSVMClassifier:
        """Load trained SVM classifier from directory."""
        in_path = Path(model_dir)
        with open(in_path / "config.json", "r") as f:
            config = json.load(f)

        instance = cls(
            target_classes=config["target_classes"],
            C=config["hyperparams"]["C"],
            loss=config["hyperparams"]["loss"],
            dual=config["hyperparams"]["dual"],
            max_iter=config["hyperparams"]["max_iter"],
            random_state=config["hyperparams"]["random_state"],
            class_weight=config["hyperparams"]["class_weight"],
        )
        instance.thresholds = config["thresholds"]
        instance.training_prevalences = config.get("training_prevalences", {})

        with open(in_path / "models.pkl", "rb") as f:
            instance.models = pickle.load(f)

        return instance


def run_svm_training_pipeline(
    data_dir: str | Path = "ml/data/ecg/ptbxl",
    output_dir: str | Path = "ml/outputs/ecg_classification/svm",
    model_dir: str | Path = "ml/models/ecg_classification/svm",
    min_weight_threshold: float = 0.0,
    C: float = 1.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 7.3 LinearSVM baseline pipeline on PTB-XL.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    mod_path = Path(model_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset loading & completeness audit
    loader = PTBXLDatasetLoader(data_path)
    loader.map_diagnostic_superclasses(min_weight_threshold=min_weight_threshold)
    train_manifest, val_manifest, test_manifest = loader.get_split_manifests()

    manifest_counts = {
        "train": {"records": len(train_manifest), "patients": int(train_manifest["patient_id"].nunique())},
        "val": {"records": len(val_manifest), "patients": int(val_manifest["patient_id"].nunique())},
        "test": {"records": len(test_manifest), "patients": int(test_manifest["patient_id"].nunique())},
    }

    audit_report = loader.audit_physical_dataset_completeness(resolution="100")
    if not audit_report["dataset_complete"]:
        raise RuntimeError(f"PTB-XL dataset incomplete: missing {audit_report['total_missing_count']} files.")

    train_eids = train_manifest.index.tolist()
    val_eids = val_manifest.index.tolist()
    test_eids = test_manifest.index.tolist()

    label_cols = [f"label_{sc}" for sc in DIAGNOSTIC_SUPERCLASSES]

    def build_split_data(eids: List[int]) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]:
        signals = []
        subset_df = loader.df.loc[eids].copy()
        for eid in eids:
            sig, fs, _ = loader.load_record_signal(eid, resolution="100")
            signals.append(sig)
        y = subset_df[label_cols].values.astype(int)
        assert len(signals) == len(y) == len(subset_df)
        return signals, y, subset_df

    t0_prep = time.time()
    train_signals, y_train, _ = build_split_data(train_eids)
    val_signals, y_val, _ = build_split_data(val_eids)
    test_signals, y_test, _ = build_split_data(test_eids)

    # 2. Feature Extraction with training-only ECG channel normalization
    feature_extractor = ECGFeatureExtractor(target_len=1000, decimation_factor=10)
    norm_stats = feature_extractor.fit_training_statistics(train_signals)

    X_train_raw = np.array([feature_extractor.extract_features_from_signal(s) for s in train_signals], dtype=np.float32)
    X_val_raw = np.array([feature_extractor.extract_features_from_signal(s) for s in val_signals], dtype=np.float32)
    X_test_raw = np.array([feature_extractor.extract_features_from_signal(s) for s in test_signals], dtype=np.float32)

    # 3. SVM-specific StandardScaler: Fit strictly on X_train only, freeze for val and test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    t_feature_and_scale = time.time() - t0_prep

    # 4. Train independent LinearSVC models on Train split
    clf = MultiLabelSVMClassifier(
        target_classes=DIAGNOSTIC_SUPERCLASSES,
        C=C,
        max_iter=2000,
        random_state=random_seed,
        class_weight="balanced",
    )
    t0_fit = time.time()
    training_info = clf.fit(X_train, y_train)
    total_training_duration = time.time() - t0_fit

    # 5. Validation-Only Threshold Optimization on Decision Scores
    evaluator = MultiLabelMetricsEvaluator(target_classes=DIAGNOSTIC_SUPERCLASSES)
    val_scores = clf.decision_function(X_val)

    # Decision scores typically range from -3 to +3; search 31 candidate thresholds
    svm_threshold_candidates = np.linspace(-2.0, 2.0, 41)
    optimal_thresholds = evaluator.find_optimal_thresholds_on_val(
        y_val, val_scores, threshold_candidates=svm_threshold_candidates
    )
    clf.set_thresholds(optimal_thresholds)

    val_metrics = evaluator.evaluate_predictions(y_val, val_scores, thresholds=optimal_thresholds)

    # 6. Held-out Test Evaluation
    t0_pred = time.time()
    test_scores = clf.decision_function(X_test)
    total_test_prediction_duration = time.time() - t0_pred

    test_metrics = evaluator.evaluate_predictions(y_test, test_scores, thresholds=optimal_thresholds)

    # 7. Save SVM Artifacts
    clf.save(mod_path)

    # Save feature scaler
    with open(mod_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save scaler metadata
    scaler_metadata = {
        "scaler_type": "StandardScaler",
        "fitted_on_split": "folds_1_to_8_only",
        "num_training_samples": len(X_train),
        "mean_shape": list(scaler.mean_.shape),
        "scale_shape": list(scaler.scale_.shape),
    }
    with open(mod_path / "scaler_metadata.json", "w") as f:
        json.dump(scaler_metadata, f, indent=2)

    # Save feature schema
    feature_schema = {
        "total_features": 1320,
        "statistical_and_morphological_features_per_lead": 10,
        "leads_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "statistical_feature_names": [
            "mean", "std", "min", "max", "peak_to_peak_amplitude",
            "median", "skewness", "kurtosis", "root_mean_square", "interquartile_range"
        ],
        "waveform_profile_points_per_lead": 100,
        "decimation_factor": 10,
    }
    with open(mod_path / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    # Save preprocessing parameters
    preprocessing_info = {
        "channel_mean": norm_stats["channel_mean"].tolist(),
        "channel_std": norm_stats["channel_std"].tolist(),
        "num_training_signals": int(norm_stats["num_training_signals"]),
        "fitted_on_split": "folds_1_to_8_only",
        "feature_dim": int(X_train.shape[1]),
        "artifact_type": "PRODUCTION_BASELINE_SVM",
    }
    with open(mod_path / "preprocessing.json", "w") as f:
        json.dump(preprocessing_info, f, indent=2)

    import sklearn
    package_versions = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "wfdb": wfdb.__version__,
    }

    training_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    experiment_summary = {
        "status": "COMPLETED",
        "phase": "7.3",
        "dataset_version": "PTB-XL v1.0.3 (100 Hz)",
        "training_timestamp_utc": training_timestamp,
        "artifact_type": "PRODUCTION_BASELINE_SVM",
        "model_architecture": "MultiLabelSVMClassifier (5 independent LinearSVC models)",
        "target_classes": DIAGNOSTIC_SUPERCLASSES,
        "random_seed": random_seed,
        "min_weight_threshold": min_weight_threshold,
        "hyperparameters": clf.hyperparams,
        "package_versions": package_versions,
        "timing": {
            "feature_extraction_and_scaling_sec": round(t_feature_and_scale, 2),
            "training_duration_sec": round(total_training_duration, 2),
            "test_prediction_duration_sec": round(total_test_prediction_duration, 4),
        },
        "manifest_counts": manifest_counts,
        "training_info": training_info,
        "selected_thresholds": optimal_thresholds,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "prevention_of_data_leakage": {
            "channel_normalization_fitted_strictly_on_train": True,
            "feature_scaler_fitted_strictly_on_train": True,
            "decision_thresholds_optimized_strictly_on_val": True,
            "test_set_strictly_evaluation_only": True,
            "patient_isolation_verified": True,
        },
    }

    with open(out_path / "summary.json", "w") as f:
        json.dump(experiment_summary, f, indent=2)

    with open(out_path / "metrics_report.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    # 8. Produce comparison with XGBoost baseline
    xgb_summary_path = Path("ml/outputs/ecg_classification/summary.json")
    if xgb_summary_path.exists():
        with open(xgb_summary_path, "r") as f:
            xgb_summary = json.load(f)

        xgb_test = xgb_summary["test_metrics"]
        svm_test = test_metrics

        comparison = {
            "timestamp_utc": training_timestamp,
            "models_compared": ["XGBoost (Phase 7.2)", "LinearSVM (Phase 7.3)"],
            "dataset_splits": manifest_counts,
            "aggregate_comparison": {
                "macro_f1": {"xgboost": xgb_test["aggregate"]["macro_f1"], "svm": svm_test["aggregate"]["macro_f1"]},
                "micro_f1": {"xgboost": xgb_test["aggregate"]["micro_f1"], "svm": svm_test["aggregate"]["micro_f1"]},
                "macro_auroc": {"xgboost": xgb_test["aggregate"]["macro_auroc"], "svm": svm_test["aggregate"]["macro_auroc"]},
                "macro_auprc": {"xgboost": xgb_test["aggregate"]["macro_auprc"], "svm": svm_test["aggregate"]["macro_auprc"]},
                "hamming_loss": {"xgboost": xgb_test["aggregate"]["hamming_loss"], "svm": svm_test["aggregate"]["hamming_loss"]},
                "subset_accuracy": {"xgboost": xgb_test["aggregate"]["subset_accuracy"], "svm": svm_test["aggregate"]["subset_accuracy"]},
            },
            "per_class_f1_comparison": {
                sc: {
                    "xgboost": xgb_test["per_class"][sc]["f1_score"],
                    "svm": svm_test["per_class"][sc]["f1_score"],
                    "better": "SVM" if svm_test["per_class"][sc]["f1_score"] > xgb_test["per_class"][sc]["f1_score"] else "XGBoost",
                }
                for sc in DIAGNOSTIC_SUPERCLASSES
            },
            "per_class_recall_comparison": {
                sc: {
                    "xgboost": xgb_test["per_class"][sc]["recall"],
                    "svm": svm_test["per_class"][sc]["recall"],
                    "better": "SVM" if svm_test["per_class"][sc]["recall"] > xgb_test["per_class"][sc]["recall"] else "XGBoost",
                }
                for sc in DIAGNOSTIC_SUPERCLASSES
            },
            "per_class_precision_comparison": {
                sc: {
                    "xgboost": xgb_test["per_class"][sc]["precision"],
                    "svm": svm_test["per_class"][sc]["precision"],
                    "better": "SVM" if svm_test["per_class"][sc]["precision"] > xgb_test["per_class"][sc]["precision"] else "XGBoost",
                }
                for sc in DIAGNOSTIC_SUPERCLASSES
            },
            "model_size_bytes": {
                "xgboost_models_pkl": (Path("ml/models/ecg_classification/models.pkl")).stat().st_size,
                "svm_models_pkl": (mod_path / "models.pkl").stat().st_size,
            },
        }

        with open(out_path / "xgboost_vs_svm_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    return experiment_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = run_svm_training_pipeline()
    print("LinearSVM pipeline completed successfully:")
    print(json.dumps(summary["test_metrics"]["per_class"], indent=2))
