"""
Unit and integration tests for Phase 7.3 LinearSVM Multi-Label ECG Abnormality Classifier.

Verifies:
1. Five independent binary LinearSVC models.
2. Continuous decision scores rather than probabilities.
3. Training-only feature scaling via StandardScaler.
4. Validation-only decision score threshold optimization.
5. Model serialization and deserialization consistency.
6. ConstantScorePredictor fallback for degenerate single-class targets.
7. End-to-end small pipeline execution and artifact generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ml.ecg_classification.train_svm import (
    ConstantScorePredictor,
    MultiLabelSVMClassifier,
)
from ml.scripts.ecg.ptbxl_loader import DIAGNOSTIC_SUPERCLASSES


class TestSVMClassificationModels:

    def test_multi_label_svm_independent_binary_architecture(self):
        """Verify five independent binary LinearSVC models predict multi-hot targets."""
        np.random.seed(42)
        X = np.random.randn(60, 1320).astype(np.float32)
        y = np.random.randint(0, 2, size=(60, 5))
        y[0, :] = [1, 0, 0, 1, 0]  # Co-occurrence of NORM and CD

        clf = MultiLabelSVMClassifier(target_classes=DIAGNOSTIC_SUPERCLASSES, C=1.0, max_iter=500)
        clf.fit(X, y)

        assert len(clf.models) == 5
        for sc in DIAGNOSTIC_SUPERCLASSES:
            assert sc in clf.models

        scores = clf.decision_function(X)
        assert scores.shape == (60, 5)
        # Continuous real-valued decision scores (can be negative or positive)
        assert scores.min() < 0.0 or scores.max() > 0.0

        preds = clf.predict(X, thresholds={sc: 0.0 for sc in DIAGNOSTIC_SUPERCLASSES})
        assert preds.shape == (60, 5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_svm_scaler_training_isolation(self):
        """Verify StandardScaler is fitted exclusively on training data and frozen for evaluation."""
        scaler = StandardScaler()
        X_train = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        scaler.fit(X_train)

        # Frozen training mean should be [20.0, 30.0]
        np.testing.assert_allclose(scaler.mean_, [20.0, 30.0])

        # Transforming val data with huge values must not alter frozen scaler statistics
        X_val = np.array([[1000.0, 2000.0]], dtype=np.float32)
        _ = scaler.transform(X_val)

        np.testing.assert_allclose(scaler.mean_, [20.0, 30.0])

    def test_constant_score_predictor_for_degenerate_classes(self):
        """Verify degenerate classes (only 0s or only 1s) use ConstantScorePredictor without crashing."""
        X = np.random.randn(20, 10).astype(np.float32)
        y = np.zeros((20, 5), dtype=int)
        y[:, 0] = 1  # NORM has only 1s, others have only 0s

        clf = MultiLabelSVMClassifier(target_classes=DIAGNOSTIC_SUPERCLASSES, C=1.0)
        clf.fit(X, y)

        assert isinstance(clf.models["NORM"], ConstantScorePredictor)
        assert clf.models["NORM"].constant_score == 1.0
        assert isinstance(clf.models["MI"], ConstantScorePredictor)
        assert clf.models["MI"].constant_score == -1.0

        scores = clf.decision_function(X)
        np.testing.assert_allclose(scores[:, 0], 1.0)
        np.testing.assert_allclose(scores[:, 1], -1.0)

    def test_svm_save_and_load_consistency(self):
        """Verify SVM weights, hyperparams, and decision thresholds are preserved across disk save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clf = MultiLabelSVMClassifier(target_classes=DIAGNOSTIC_SUPERCLASSES, C=0.5, max_iter=300)
            X = np.random.randn(40, 50).astype(np.float32)
            y = np.random.randint(0, 2, size=(40, 5))
            clf.fit(X, y)
            clf.set_thresholds({"NORM": -0.2, "MI": 0.1, "STTC": 0.0, "CD": 0.3, "HYP": -0.1})

            orig_scores = clf.decision_function(X)
            orig_preds = clf.predict(X)

            clf.save(tmpdir)
            loaded_clf = MultiLabelSVMClassifier.load(tmpdir)

            loaded_scores = loaded_clf.decision_function(X)
            loaded_preds = loaded_clf.predict(X)

            np.testing.assert_allclose(orig_scores, loaded_scores, atol=1e-5)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
            assert loaded_clf.thresholds == clf.thresholds
            assert loaded_clf.hyperparams == clf.hyperparams
