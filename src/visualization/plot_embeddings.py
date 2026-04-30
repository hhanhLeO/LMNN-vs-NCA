"""
src/visualization/plot_embeddings.py

2-D scatter plots of learned embeddings from NCA and LMNN,
compared side-by-side with PCA and raw Euclidean coordinates.

plot_embeddings(dataset, models, save_path)
    Produces a 1×4 panel: Raw | PCA | NCA | LMNN (all in 2D).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# Colour ramps for up to 10 classes
_CLASS_COLOURS = [
    "#3266ad", "#1D9E75", "#e07b39",
    "#9b59b6", "#c0392b", "#2ecc71",
    "#e67e22", "#1abc9c", "#e74c3c", "#34495e",
]


def _scatter(ax, X2d, y, title, class_names=None):
    """Draw a 2-D scatter coloured by class label."""
    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        label = class_names[cls] if class_names else str(cls)
        ax.scatter(
            X2d[mask, 0], X2d[mask, 1],
            s=18, alpha=0.65,
            c=_CLASS_COLOURS[cls % len(_CLASS_COLOURS)],
            label=label, edgecolors="none",
        )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#dddddd")


def plot_embeddings(
    dataset: dict,
    nca_model=None,
    lmnn_model=None,
    save_path: str = "results/plots/embeddings.png",
    random_state: int = 42,
    test_size: float = 0.25,
) -> None:
    """
    Four-panel embedding comparison:
        [Raw 2PC] [PCA] [NCA] [LMNN]

    If a model is None, its panel is skipped (only 3 panels shown).
    Models are fit on the train split; all points projected and shown.

    Parameters
    ----------
    dataset    : dict  standard dataset dict (must have X, y, name)
    nca_model  : NCAModel or None
    lmnn_model : LMNNModel or None
    save_path  : str   output PNG path
    random_state : int
    test_size  : float
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X, y = dataset["X"], dataset["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    sc = StandardScaler()
    X_all_s = sc.fit_transform(X)  # full scaled set for display
    X_tr_s  = sc.transform(X_tr)

    panels = []

    # Panel 1: Raw (first 2 standardised dims)
    panels.append((X_all_s[:, :2], y, "Raw (first 2 dims)"))

    # Panel 2: PCA
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(X_tr_s)
    X_pca = pca.transform(X_all_s)
    panels.append((X_pca, y, "PCA (2 components)"))

    # Panel 3: NCA
    if nca_model is not None:
        nca_model.fit(X_tr, y_tr)
        X_nca_2d = nca_model._nca.transform(nca_model._scaler.transform(X))
        if X_nca_2d.shape[1] > 2:
            X_nca_2d = X_nca_2d[:, :2]
        panels.append((X_nca_2d, y, "NCA embedding"))

    # Panel 4: LMNN
    if lmnn_model is not None:
        lmnn_model.fit(X_tr, y_tr)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_lmnn_2d = lmnn_model._lmnn.transform(lmnn_model._scaler.transform(X))
        if X_lmnn_2d.shape[1] > 2:
            X_lmnn_2d = X_lmnn_2d[:, :2]
        panels.append((X_lmnn_2d, y, "LMNN embedding"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    if n == 1:
        axes = [axes]

    class_names = dataset.get("class_names")
    for ax, (X2d, y_, title) in zip(axes, panels):
        _scatter(ax, X2d, y_, title, class_names)

    # Shared legend
    classes  = np.unique(y)
    patches  = [
        mpatches.Patch(
            color=_CLASS_COLOURS[c % len(_CLASS_COLOURS)],
            label=class_names[c] if class_names else str(c),
        )
        for c in classes
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=len(classes), fontsize=8.5, frameon=False,
    )

    fig.suptitle(f"Embedding Comparison — {dataset['name']}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")