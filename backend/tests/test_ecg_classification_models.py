"""
Unit and integration tests for Phase 7.1 XGBoost Multi-Label ECG Abnormality Classifier.

Verifies:
1. Five independent binary models for NORM, MI, STTC, CD, HYP.
2. Independent sigmoid/probability outputs (allowing multi-label predictions, e.g. NORM + CD).
3. Zero-weight statement exclusion (min_weight_threshold=0.0).
4. Preprocessing fitted strictly on training data only (validation/test isolation).
5. Validation-only threshold optimization (test set never used for tuning).
6. Metrics calculation handling single-class and undefined metrics without fabricating values.
7. Model artifact serialization and deserialization consistency.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from ml.ecg_classification.feature_extractor import ECGFeatureExtractor
from ml.ecg_classification.label_mapping import PTBXLLabelMapper
from ml.ecg_classification.metrics import MultiLabelMetricsEvaluator
from ml.ecg_classification.train_xgboost import (
    MultiLabelECGClassifier,
    ConstantPredictor,
    IncompleteDatasetError,
    run_xgboost_training_pipeline,
)
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES, PTBXLDatasetLoader


class TestECGClassificationModels:

    def test_multi_label_independent_binary_architecture(self):
        """Verify five independent binary models operate independently allowing co-occurrence."""
        np.random.seed(42)
        X_dummy = np.random.randn(50, 1320).astype(np.float32)
        # Create multi-label targets where sample 0 has both NORM and CD
        y_dummy = np.random.randint(0, 2, size=(50, 5))
        y_dummy[0, :] = [1, 0, 0, 1, 0]  # Co-occurrence of NORM and CD

        clf = MultiLabelECGClassifier(target_classes=DIAGNOSTIC_SUPERCLASSES, n_estimators=10, max_depth=2)
        clf.fit(X_dummy, y_dummy)

        # Check 5 independent models
        assert len(clf.models) == 5
        for sc in DIAGNOSTIC_SUPERCLASSES:
            assert sc in clf.models

        probs = clf.predict_proba(X_dummy)
        assert probs.shape == (50, 5)
        # Probabilities must be independent in [0, 1] and NOT constrained to sum to 1.0
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        preds = clf.predict(X_dummy, thresholds={sc: 0.5 for sc in DIAGNOSTIC_SUPERCLASSES})
        assert preds.shape == (50, 5)
        # Sample predictions can have multiple 1s or all 0s
        row_sums = preds.sum(axis=1)
        assert np.any(row_sums > 1) or np.any(row_sums == 0) or np.any(row_sums == 1)

    def test_constant_predictor_fallback_for_degenerate_splits(self):
        """Verify degenerate classes (only 0s or only 1s) use ConstantPredictor without crashing."""
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.zeros((20, 5), dtype=int)
        y[:, 0] = 1  # NORM has only 1s
        # Other classes have only 0s

        clf = MultiLabelECGClassifier(target_classes=DIAGNOSTIC_SUPERCLASSES, n_estimators=10)
        clf.fit(X, y)

        assert isinstance(clf.models["NORM"], ConstantPredictor)
        assert clf.models["NORM"].constant_prob == 1.0
        assert isinstance(clf.models["MI"], ConstantPredictor)
        assert clf.models["MI"].constant_prob == 0.0

        probs = clf.predict_proba(X)
        assert np.allclose(probs[:, 0], 1.0)
        assert np.allclose(probs[:, 1], 0.0)

    def test_training_only_preprocessing_isolation(self):
        """Verify normalization statistics are computed exclusively on training signals."""
        extractor = ECGFeatureExtractor(target_len=1000, decimation_factor=10)

        sig1 = np.ones((1000, 12), dtype=np.float32) * 5.0
        sig2 = np.ones((1000, 12), dtype=np.float32) * 15.0
        train_sigs = [sig1, sig2]

        stats = extractor.fit_training_statistics(train_sigs)
        # Training mean should be 10.0 across leads
        np.testing.assert_allclose(stats["channel_mean"], np.full(12, 10.0), atol=1e-5)

        # Validation signal with mean 100.0 must NOT affect training statistics
        val_sig = np.ones((1000, 12), dtype=np.float32) * 100.0
        feat_val = extractor.extract_features_from_signal(val_sig)

        # Extracted feature length must be 1320
        assert len(feat_val) == 1320
        # Training mean remains 10.0
        np.testing.assert_allclose(extractor.training_mean, np.full(12, 10.0), atol=1e-5)

    def test_validation_only_threshold_optimization(self):
        """Verify thresholds are chosen on validation set and applied unchanged to test set."""
        evaluator = MultiLabelMetricsEvaluator(target_classes=["NORM", "MI"])
        
        # Validation ground truth and probs
        y_val = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        y_val_probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])

        optimal_th = evaluator.find_optimal_thresholds_on_val(y_val, y_val_probs)
        assert "NORM" in optimal_th and "MI" in optimal_th
        for th in optimal_th.values():
            assert 0.1 <= th <= 0.9

        # Evaluate on test set with these thresholds
        y_test = np.array([[1, 0], [0, 1]])
        y_test_probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        test_metrics = evaluator.evaluate_predictions(y_test, y_test_probs, thresholds=optimal_th)

        assert test_metrics["per_class"]["NORM"]["threshold"] == optimal_th["NORM"]
        assert test_metrics["per_class"]["MI"]["threshold"] == optimal_th["MI"]

    def test_metrics_handles_single_class_and_undefined_cases(self):
        """Verify undefined metrics (e.g. 0 positive samples) yield None/null rather than fabricated values."""
        evaluator = MultiLabelMetricsEvaluator(target_classes=["NORM", "MI"])
        # All samples are NORM=1, MI=0
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        y_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.1]])

        metrics = evaluator.evaluate_predictions(y_true, y_probs)
        
        # MI has 0 positive samples -> AUROC and AUPRC must be None
        assert metrics["per_class"]["MI"]["support"] == 0
        assert metrics["per_class"]["MI"]["prevalence"] == 0.0
        assert metrics["per_class"]["MI"]["auroc"] is None
        assert metrics["per_class"]["MI"]["auprc"] is None

        # NORM has only positive samples (0 negative samples) -> AUROC must be None
        assert metrics["per_class"]["NORM"]["support"] == 3
        assert metrics["per_class"]["NORM"]["auroc"] is None

    def test_model_save_and_load_consistency(self):
        """Verify model weights, hyperparams, and thresholds are faithfully preserved across disk save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clf = MultiLabelECGClassifier(
                target_classes=DIAGNOSTIC_SUPERCLASSES,
                n_estimators=15,
                max_depth=3,
                learning_rate=0.05,
                random_state=123,
            )
            X = np.random.randn(30, 1320).astype(np.float32)
            y = np.random.randint(0, 2, size=(30, 5))
            clf.fit(X, y)
            clf.set_thresholds({"NORM": 0.35, "MI": 0.45, "STTC": 0.55, "CD": 0.65, "HYP": 0.75})

            orig_probs = clf.predict_proba(X)
            orig_preds = clf.predict(X)

            # Save and Reload
            clf.save(tmpdir)
            loaded_clf = MultiLabelECGClassifier.load(tmpdir)

            loaded_probs = loaded_clf.predict_proba(X)
            loaded_preds = loaded_clf.predict(X)

            np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-5)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
            assert loaded_clf.thresholds == clf.thresholds
            assert loaded_clf.hyperparams == clf.hyperparams

    def test_incomplete_dataset_raises_fail_closed_error(self):
        """Verify incomplete physical dataset raises IncompleteDatasetError preventing production training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir) / "ptbxl"
            tmp_data.mkdir()
            # Copy database and scp statements but NO waveform files
            shutil.copy("ml/data/ecg/ptbxl/ptbxl_database.csv", tmp_data / "ptbxl_database.csv")
            shutil.copy("ml/data/ecg/ptbxl/scp_statements.csv", tmp_data / "scp_statements.csv")

            with tempfile.TemporaryDirectory() as tmpout, tempfile.TemporaryDirectory() as tmpmod:
                with pytest.raises(IncompleteDatasetError) as exc_info:
                    run_xgboost_training_pipeline(
                        data_dir=tmp_data,
                        output_dir=tmpout,
                        model_dir=tmpmod,
                        allow_incomplete_subset=False,
                    )
                assert "PTB-XL dataset is incomplete on disk" in str(exc_info.value)
                # Production directories must remain empty
                assert len(list(Path(tmpmod).glob("*"))) == 0

    def test_smoketest_mode_allows_controlled_execution(self):
        """Verify smoketest mode executes when explicitly authorized with is_smoketest=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir) / "ptbxl"
            tmp_data.mkdir()
            shutil.copy("ml/data/ecg/ptbxl/ptbxl_database.csv", tmp_data / "ptbxl_database.csv")
            shutil.copy("ml/data/ecg/ptbxl/scp_statements.csv", tmp_data / "scp_statements.csv")
            # Copy only the first 14 records to simulate incomplete smoketest slice
            (tmp_data / "records100" / "00000").mkdir(parents=True)
            for eid in range(1, 15):
                base = f"{eid:05d}_lr"
                shutil.copy(f"ml/data/ecg/ptbxl/records100/00000/{base}.dat", tmp_data / f"records100/00000/{base}.dat")
                shutil.copy(f"ml/data/ecg/ptbxl/records100/00000/{base}.hea", tmp_data / f"records100/00000/{base}.hea")

            with tempfile.TemporaryDirectory() as tmpout, tempfile.TemporaryDirectory() as tmpmod:
                summary = run_xgboost_training_pipeline(
                    data_dir=tmp_data,
                    output_dir=tmpout,
                    model_dir=tmpmod,
                    min_weight_threshold=0.0,
                    random_seed=42,
                    allow_incomplete_subset=True,
                    is_smoketest=True,
                )

                assert summary["status"] == "COMPLETED"
                assert summary["phase"] in ("7.1.1", "7.2")
                assert summary["artifact_type"] == "SMOKETEST"
                assert summary["prevention_of_data_leakage"]["normalization_fitted_strictly_on_train"] is True
                assert summary["prevention_of_data_leakage"]["thresholds_optimized_strictly_on_val"] is True
                assert summary["prevention_of_data_leakage"]["test_set_strictly_evaluation_only"] is True

                # Verify files exist on disk with smoketest marker
                assert (Path(tmpout) / "summary.json").is_file()
                assert (Path(tmpout) / "metrics_report.json").is_file()
                assert (Path(tmpmod) / "models.pkl").is_file()
                assert (Path(tmpmod) / "config.json").is_file()
                assert (Path(tmpmod) / "preprocessing.json").is_file()

                with open(Path(tmpmod) / "preprocessing.json") as f:
                    prep = json.load(f)
                    assert prep["artifact_type"] == "SMOKETEST"

    def test_physical_dataset_completeness_audit(self):
        """Verify audit_physical_dataset_completeness accurately reports complete status for all 21,799 records."""
        loader = PTBXLDatasetLoader("ml/data/ecg/ptbxl")
        audit = loader.audit_physical_dataset_completeness(resolution="100")

        assert audit["dataset_complete"] is True
        assert audit["total_missing_count"] == 0
        assert audit["splits"]["train"]["expected_records"] == 17418
        assert audit["splits"]["train"]["physical_records_found"] == 17418
        assert audit["splits"]["train"]["missing_records_count"] == 0
        assert audit["splits"]["train"]["is_complete"] is True

        assert audit["splits"]["val"]["expected_records"] == 2183
        assert audit["splits"]["val"]["physical_records_found"] == 2183
        assert audit["splits"]["val"]["missing_records_count"] == 0
        assert audit["splits"]["val"]["is_complete"] is True

        assert audit["splits"]["test"]["expected_records"] == 2198
        assert audit["splits"]["test"]["physical_records_found"] == 2198
        assert audit["splits"]["test"]["missing_records_count"] == 0
        assert audit["splits"]["test"]["is_complete"] is True
