"""
src/models/lmnn_model.py

Large Margin Nearest Neighbor (LMNN) + KNN classifier.

LMNN (Weinberger & Saul, 2009) learns a Mahalanobis distance metric M = L^T L
(equivalently a linear transformation L) by solving a semi-definite program
with two objectives:

  1. Pull: keep k "target neighbours" (same-class points) close in L-space.
  2. Push: ensure impostors (different-class points within the margin) are
           pushed at least 1 unit outside the target-neighbour ball.

The combined loss is:
    Σ_{i,j∈N(i)} ||L(x_i - x_j)||²
  + λ Σ_{i,j∈N(i),l: y_l≠y_i} max(0, 1 + ||L(x_i-x_j)||² - ||L(x_i-x_l)||²)

where N(i) is the set of k target neighbours of point i.

Key properties:
  • Explicit large-margin objective — robust to outliers.
  • Uses a global metric (one L for all classes), unlike per-class methods.
  • Often converges faster than NCA on large datasets.
  • May underfit per-class anisotropy (single global metric assumption).

Requires: metric-learn == 0.7.0  with  scikit-learn == 1.5.2
"""

import warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from metric_learn import LMNN as _LMNN
    _LMNN_AVAILABLE = True
except ImportError:
    _LMNN_AVAILABLE = False


class LMNNModel:
    """
    LMNN metric learning followed by KNN classification.

    Pipeline: StandardScaler → LMNN (fit on train) → KNeighborsClassifier.
    transform() returns coordinates in the learned L-space.

    Raises ImportError at instantiation if metric-learn is not installed.
    """

    name = "LMNN"

    def __init__(
        self,
        k: int = 5,
        n_components: int = 6,
        max_iter: int = 150,
        random_state: int = 42,
    ):
        if not _LMNN_AVAILABLE:
            raise ImportError(
                "metric-learn is required for LMNNModel. "
                "Install it with:  pip install metric-learn==0.7.0"
            )

        self.k            = k
        self.n_components = n_components
        self.max_iter     = max_iter
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._lmnn   = _LMNN(
            n_components=n_components,
            k=k,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._knn    = KNeighborsClassifier(n_neighbors=k)
        self._fitted = False

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LMNNModel":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled   = self._scaler.fit_transform(X_train)
            X_embedded = self._lmnn.fit_transform(X_scaled, y_train)
        self._knn.fit(X_embedded, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._knn.predict(self.transform(X_test))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the learned LMNN (L-space) embedding."""
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._lmnn.transform(self._scaler.transform(X))

    def get_params(self) -> dict:
        return {
            "k":            self.k,
            "n_components": self.n_components,
            "max_iter":     self.max_iter,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return (
            f"LMNNModel(k={self.k}, "
            f"n_components={self.n_components}, "
            f"max_iter={self.max_iter})"
        )