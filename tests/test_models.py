"""tests/test_models.py — Unit tests for all three model classes."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.datasets import load_iris

from src.models import KNNBaseline, NCAModel, LMNNModel, ALL_MODELS

X_FULL, Y_FULL = load_iris(return_X_y=True)
X_TR, X_TE = X_FULL[:120], X_FULL[120:]
Y_TR, Y_TE = Y_FULL[:120], Y_FULL[120:]


# ── Shared interface contract ─────────────────────────────────────────────────

@pytest.mark.parametrize("ModelCls,kwargs", [
    (KNNBaseline, {"k": 3}),
    (NCAModel,    {"k": 3, "n_components": 2, "max_iter": 50}),
    (LMNNModel,   {"k": 3, "n_components": 2, "max_iter": 50}),
])
def test_fit_predict_shape(ModelCls, kwargs):
    m = ModelCls(**kwargs)
    m.fit(X_TR, Y_TR)
    preds = m.predict(X_TE)
    assert preds.shape == (len(X_TE),)
    assert set(preds).issubset(set(Y_TR))


@pytest.mark.parametrize("ModelCls,kwargs", [
    (KNNBaseline, {"k": 3}),
    (NCAModel,    {"k": 3, "n_components": 2, "max_iter": 50}),
    (LMNNModel,   {"k": 3, "n_components": 2, "max_iter": 50}),
])
def test_transform_shape(ModelCls, kwargs):
    m = ModelCls(**kwargs)
    m.fit(X_TR, Y_TR)
    emb = m.transform(X_TE)
    assert emb.ndim == 2
    assert emb.shape[0] == len(X_TE)


@pytest.mark.parametrize("ModelCls,kwargs", [
    (KNNBaseline, {"k": 3}),
    (NCAModel,    {"k": 3, "n_components": 2, "max_iter": 50}),
    (LMNNModel,   {"k": 3, "n_components": 2, "max_iter": 50}),
])
def test_get_params_returns_dict(ModelCls, kwargs):
    m = ModelCls(**kwargs)
    p = m.get_params()
    assert isinstance(p, dict)
    assert "k" in p


@pytest.mark.parametrize("ModelCls,kwargs", [
    (KNNBaseline, {}),
    (NCAModel,    {"max_iter": 50}),
    (LMNNModel,   {"max_iter": 50}),
])
def test_predict_before_fit_raises(ModelCls, kwargs):
    m = ModelCls(**kwargs)
    with pytest.raises(RuntimeError, match="fitted"):
        m.predict(X_TE)


# ── NCA dimensionality ────────────────────────────────────────────────────────

def test_nca_reduces_dimensions():
    m = NCAModel(k=3, n_components=2, max_iter=50)
    m.fit(X_TR, Y_TR)
    assert m.transform(X_TE).shape[1] == 2


# ── LMNN dimensionality ───────────────────────────────────────────────────────

def test_lmnn_reduces_dimensions():
    m = LMNNModel(k=3, n_components=2, max_iter=50)
    m.fit(X_TR, Y_TR)
    assert m.transform(X_TE).shape[1] == 2


# ── Registry ─────────────────────────────────────────────────────────────────

def test_all_models_in_registry():
    assert set(ALL_MODELS.keys()) == {"KNN", "NCA", "LMNN"}


# ── Accuracy sanity check ─────────────────────────────────────────────────────

def test_knn_accuracy_above_chance():
    from sklearn.metrics import accuracy_score
    m = KNNBaseline(k=5)
    m.fit(X_TR, Y_TR)
    acc = accuracy_score(Y_TE, m.predict(X_TE))
    assert acc > 1 / 3, f"KNN accuracy {acc:.3f} is at chance level"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])