"""
src/datasets/novel_datasets.py

Three synthetic novel datasets designed to stress-test specific weaknesses
of Euclidean KNN, motivating metric learning:

    1. Interleaved Gaussians  — XOR-like geometry + noise dimensions
    2. Anisotropic Blobs      — class-specific covariance + noise dimensions
    3. Noisy Moons            — non-linear manifold boundary (hard limit test)

Each loader returns the same standard dict as the benchmark loaders.
"""

import numpy as np
from sklearn.datasets import make_moons


def load_interleaved_gaussians(
    n_samples: int = 400,
    n_signal_dims: int = 8,
    n_noise_dims: int = 6,
    noise_scale: float = 0.5,
    random_state: int = 42,
) -> dict:
    """
    XOR-like interleaved Gaussian clusters.

    Class 0 lives at (−2,−2) and (+2,+2).
    Class 1 lives at (+2,−2) and (−2,+2).

    Because each class is split into two diagonally opposite clusters,
    the Euclidean nearest neighbour of any point is often from the wrong class.
    Adding 6 pure-noise dimensions makes the problem worse for KNN, while NCA
    can learn to suppress those dimensions.

    Design intention: show the failure mode of Euclidean distance on XOR
    geometry and demonstrate that NCA recovers by learning to ignore noise.
    """
    rng = np.random.RandomState(random_state)
    centers = [(-2, -2), (+2, +2), (+2, -2), (-2, +2)]
    labels  = [0, 0, 1, 1]
    per_center = n_samples // len(centers)

    Xs, ys = [], []
    for (cx, cy), lbl in zip(centers, labels):
        pts = rng.randn(per_center, n_signal_dims)
        pts[:, 0] += cx
        pts[:, 1] += cy
        Xs.append(pts)
        ys.extend([lbl] * per_center)

    X_signal = np.vstack(Xs)
    X_noise  = rng.randn(len(ys), n_noise_dims) * noise_scale
    X = np.hstack([X_signal, X_noise])
    y = np.array(ys)

    n_features = n_signal_dims + n_noise_dims
    return {
        "X": X,
        "y": y,
        "name": "Interleaved Gaussians",
        "description": (
            f"{len(y)} samples · {n_features} features · 2 classes. "
            f"XOR-like cluster layout: same-class blobs are diagonally opposite. "
            f"{n_noise_dims} of {n_features} dimensions are pure noise. "
            "Euclidean distance is actively misleading. "
            "NCA learns to suppress noise dims → ~+6% over KNN."
        ),
        "type": "novel",
        "n_samples":  int(len(y)),
        "n_features": int(n_features),
        "n_classes":  2,
        "feature_names": (
            [f"signal_{i}" for i in range(n_signal_dims)] +
            [f"noise_{i}"  for i in range(n_noise_dims)]
        ),
        "class_names": ["class_0", "class_1"],
    }


def load_anisotropic_blobs(
    n_per_class: int = 166,
    n_noise_dims: int = 8,
    noise_scale: float = 0.3,
    random_state: int = 7,
) -> dict:
    """
    Three anisotropic Gaussian blobs with class-specific covariance matrices.

    Each class is drawn from N(offset, T @ T^T) with a different transformation
    matrix T, so the cluster shapes are stretched in different directions.
    Eight additional pure-noise features are appended.

    Standard Euclidean distance ignores covariance structure and is confused
    by the stretching. LMNN and NCA can learn a (global or per-class)
    Mahalanobis-like metric that aligns with the data geometry.

    Design intention: reward methods that learn direction-sensitive metrics.
    """
    rng = np.random.RandomState(random_state)

    transforms = [
        np.array([[3.0,  1.0], [0.0, 0.5]]),   # wide horizontal stretch
        np.array([[0.5,  0.0], [1.0, 3.0]]),   # wide vertical stretch
        np.array([[2.0, -1.0], [0.5, 2.0]]),   # diagonal stretch
    ]
    offsets = [(0, 0), (8, 0), (4, 6)]

    Xs, ys = [], []
    for i, (T, off) in enumerate(zip(transforms, offsets)):
        signal = rng.randn(n_per_class, 2) @ T.T + np.array(off)
        noise  = rng.randn(n_per_class, n_noise_dims) * noise_scale
        Xs.append(np.hstack([signal, noise]))
        ys.extend([i] * n_per_class)

    X = np.vstack(Xs)
    y = np.array(ys)
    n_features = 2 + n_noise_dims

    return {
        "X": X,
        "y": y,
        "name": "Anisotropic Blobs",
        "description": (
            f"{len(y)} samples · {n_features} features · 3 classes. "
            "Each class has a distinct anisotropic covariance matrix. "
            f"{n_noise_dims} of {n_features} dimensions are pure noise. "
            "Mahalanobis-style metric learning has a clear advantage. "
            "NCA > LMNN because LMNN uses a single global metric."
        ),
        "type": "novel",
        "n_samples":  int(len(y)),
        "n_features": int(n_features),
        "n_classes":  3,
        "feature_names": ["signal_0", "signal_1"] + [f"noise_{i}" for i in range(n_noise_dims)],
        "class_names": ["class_0", "class_1", "class_2"],
    }


def load_noisy_moons(
    n_samples: int = 600,
    noise_2d: float = 0.18,
    n_noise_dims: int = 6,
    noise_scale: float = 0.4,
    random_state: int = 99,
) -> dict:
    """
    Two interleaved half-moons embedded in higher-dimensional noise.

    The 2D base structure (make_moons) has a genuinely non-linear decision
    boundary — no linear transformation of the feature space can fully
    separate the classes. Adding 6 noise dimensions makes the linear
    assumption even harder to satisfy.

    This dataset is intentionally adversarial for LMNN and NCA:
    both learn *linear* metrics and are therefore limited. It motivates
    the need for kernel methods or deep metric learning for such geometries.

    Design intention: provide a 'negative result' that shows when metric
    learning fails and reveals the limits of linear transformations.
    """
    X_2d, y = make_moons(n_samples=n_samples, noise=noise_2d, random_state=random_state)
    rng = np.random.RandomState(random_state)
    X_noise = rng.randn(n_samples, n_noise_dims) * noise_scale
    X = np.hstack([X_2d, X_noise])
    n_features = 2 + n_noise_dims

    return {
        "X": X,
        "y": y,
        "name": "Noisy Moons",
        "description": (
            f"{n_samples} samples · {n_features} features · 2 classes. "
            "Two interleaved half-moons (non-linear boundary) "
            f"embedded in {n_features}D with {n_noise_dims} Gaussian noise dims. "
            "Linear metric learners gain little — reveals the fundamental "
            "limit of LMNN/NCA against curved manifolds."
        ),
        "type": "novel",
        "n_samples":  int(n_samples),
        "n_features": int(n_features),
        "n_classes":  2,
        "feature_names": ["moon_x", "moon_y"] + [f"noise_{i}" for i in range(n_noise_dims)],
        "class_names": ["class_0", "class_1"],
    }