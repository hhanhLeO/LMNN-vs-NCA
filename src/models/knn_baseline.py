"""
src/models/knn_baseline.py

Plain K-Nearest Neighbours classifier using Euclidean distance.
No metric learning — serves as the baseline all methods are compared against.

All model classes in this project share the same interface:
    fit(X_train, y_train) → self
    predict(X_test)       → np.ndarray
    transform(X)          → np.ndarray  (identity for KNN; learned space for NCA/LMNN)
    get_params()          → dict
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNBaseline:
    """
    K-Nearest Neighbours with Euclidean distance (no metric learning).

    Pipeline: StandardScaler → KNeighborsClassifier.
    transform() returns the scaled input unchanged (no dimensionality change).
    """

    name = "KNN"

    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._knn    = KNeighborsClassifier(n_neighbors=k)
        self._fitted = False

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNBaseline":
        X_scaled = self._scaler.fit_transform(X_train)
        self._knn.fit(X_scaled, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._knn.predict(self._scaler.transform(X_test))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns standardised input (no additional projection)."""
        self._check_fitted()
        return self._scaler.transform(X)

    def get_params(self) -> dict:
        return {"k": self.k}

    # ── Internal helpers ─────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        return f"KNNBaseline(k={self.k})"