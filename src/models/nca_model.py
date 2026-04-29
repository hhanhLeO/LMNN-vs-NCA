"""
src/models/nca_model.py

Neighborhood Component Analysis (NCA) + KNN classifier.

NCA (Goldberger et al., 2004) learns a linear transformation A of the input
space by directly maximising a soft leave-one-out KNN accuracy on the training
set. The objective is:

    max_A  Σ_i  Σ_{j: same class as i}  p_{ij}

where p_{ij} = softmax(-||A(x_i - x_j)||²) over all j ≠ i.

After training, points are projected into the learned A-space and a standard
KNN classifier is applied.

Key properties:
  • Optimises directly for KNN — very well-aligned objective.
  • Can suppress irrelevant / noisy dimensions.
  • Slower to train than LMNN for large datasets.
  • Sensitive to learning rate and initialisation.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler


class NCAModel:
    """
    NCA metric learning followed by KNN classification.

    Pipeline: StandardScaler → NCA (fit on train) → KNeighborsClassifier.
    transform() returns coordinates in the learned NCA embedding space.
    """

    name = "NCA"

    def __init__(
        self,
        k: int = 5,
        n_components: int = 6,
        max_iter: int = 150,
        random_state: int = 42,
    ):
        self.k            = k
        self.n_components = n_components
        self.max_iter     = max_iter
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._nca    = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._knn    = KNeighborsClassifier(n_neighbors=k)
        self._fitted = False

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NCAModel":
        X_scaled     = self._scaler.fit_transform(X_train)
        X_embedded   = self._nca.fit_transform(X_scaled, y_train)
        self._knn.fit(X_embedded, y_train)
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._knn.predict(self.transform(X_test))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the learned NCA embedding space."""
        self._check_fitted()
        return self._nca.transform(self._scaler.transform(X))

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
            f"NCAModel(k={self.k}, "
            f"n_components={self.n_components}, "
            f"max_iter={self.max_iter})"
        )