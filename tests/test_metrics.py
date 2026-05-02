"""tests/test_metrics.py — Unit tests for metrics and evaluator."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.datasets import load_iris

from src.evaluation.metrics import compute_metrics
from src.evaluation.evaluator import Evaluator
from src.models import KNNBaseline, NCAModel
from src.datasets import load_dataset


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y)
    assert m["accuracy"]  == 1.0
    assert m["f1_macro"]  == 1.0
    assert all(v == 1.0 for v in m["f1_per_class"])


def test_all_wrong_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0


def test_returns_required_keys():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y)
    for key in ("accuracy", "f1_macro", "f1_per_class", "confusion_matrix"):
        assert key in m


def test_confusion_matrix_shape():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = compute_metrics(y, y)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (3, 3)


def test_confusion_matrix_diagonal_perfect():
    y = np.array([0, 1, 2])
    m = compute_metrics(y, y)
    cm = np.array(m["confusion_matrix"])
    assert np.all(cm == np.eye(3))


# ── Evaluator ─────────────────────────────────────────────────────────────────

def test_evaluator_returns_all_methods():
    ds = load_dataset("iris")
    ev = Evaluator(test_size=0.25, random_state=42)
    ev.set_default_models(k=5, n_components=3, max_iter=50)
    result = ev.evaluate_dataset(ds)
    assert set(result.keys()) == {"KNN", "NCA", "LMNN"}


def test_evaluator_metrics_in_range():
    ds = load_dataset("iris")
    ev = Evaluator(test_size=0.25, random_state=42)
    ev.set_default_models(k=5, n_components=3, max_iter=50)
    result = ev.evaluate_dataset(ds)
    for method, r in result.items():
        if r["accuracy"] is not None:
            assert 0.0 <= r["accuracy"] <= 1.0, f"{method} accuracy out of range"
            assert 0.0 <= r["f1_macro"]  <= 1.0, f"{method} F1 out of range"


def test_evaluator_knn_iris_above_90pct():
    ds = load_dataset("iris")
    ev = Evaluator(test_size=0.25, random_state=42)
    ev.add_model(KNNBaseline(k=5))
    result = ev.evaluate_dataset(ds)
    assert result["KNN"]["accuracy"] >= 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])