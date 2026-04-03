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


class NCA:
    """
    NCA metric learning followed by KNN classification.

    Pipeline: StandardScaler → NCA (fit on train) → KNeighborsClassifier.
    transform() returns coordinates in the learned NCA embedding space.
    """

    name = "NCA"

    def __init__(self, k: int = 5, n_components: int = 6, max_iter: int = 150, random_state: int = 42):
        self._scaler = StandardScaler()
        self._nca = NeighborhoodComponentsAnalysis(n_components=n_components, max_iter=max_iter, random_state=random_state)
        self._knn = KNeighborsClassifier(n_neighbors=k)
        self._fitted = False

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NCA":
        X_scaled = self._scaler.fit_transform(X_train)
        X_transformed = self._nca.fit_transform(X_scaled, y_train)
        self._knn.fit(X_transformed, y_train)
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
            f"NCA(k={self.k}, "
            f"n_components={self.n_components}, "
            f"max_iter={self.max_iter})"
        )
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNN

if __name__ == "__main__":
    data = datasets.load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Using only KNN
    knn = KNN()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy score of KNN:   {acc_knn * 100:.2f}%")

    # Using KNN with NCA
    nca = NCA()
    nca.fit(X_train, y_train)
    y_pred_nca = nca.predict(X_test)
    acc_nca = accuracy_score(y_test, y_pred_nca)
    print(f"Accuracy score of NCA+KNN:   {acc_nca * 100:.2f}%")