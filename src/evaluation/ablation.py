"""
src/evaluation/ablation.py

Ablation study runner.

Sweeps a single hyperparameter (k, n_components, or max_iter) across a
list of values on a chosen dataset, recording metrics for every method
at each setting.

Usage
-----
    from src.evaluation import AblationStudy
    from src.datasets import load_dataset

    study = AblationStudy(dataset=load_dataset("iris"))

    # Sweep k from 1 to 11
    results = study.sweep(
        param="k",
        values=[1, 3, 5, 7, 9, 11],
        fixed={"n_components": 3, "max_iter": 100},
    )
    # results: {1: {"KNN": {...}, "NCA": {...}, "LMNN": {...}}, 3: ...}
"""

import time
import traceback
import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_metrics
from src.models import KNNBaseline, NCAModel, LMNNModel


_MODEL_PARAMS = {
    "KNN":  {"k"},
    "NCA":  {"k", "n_components", "max_iter"},
    "LMNN": {"k", "n_components", "max_iter"},
}


class AblationStudy:
    """
    Sweeps one hyperparameter across a list of values and
    records test metrics for KNN, NCA, and LMNN at each setting.
    """

    def __init__(
        self,
        dataset: dict,
        test_size: float = 0.25,
        random_state: int = 42,
    ):
        self.dataset      = dataset
        self.test_size    = test_size
        self.random_state = random_state

        X, y = dataset["X"], dataset["y"]
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # ── Main API ─────────────────────────────────────────────────────

    def sweep(
        self,
        param: str,
        values: list,
        fixed: dict | None = None,
    ) -> dict:
        """
        Run the sweep.

        Parameters
        ----------
        param  : str   — hyperparameter to sweep ('k', 'n_components', 'max_iter')
        values : list  — values to test
        fixed  : dict  — other hyperparameters held constant

        Returns
        -------
        dict  param_value → {method_name → metrics_dict}
        """
        fixed = fixed or {}
        results = {}

        for val in values:
            all_params = {param: val, **fixed}
            results[val] = self._run_one(all_params)

        return results

    # ── Internal ─────────────────────────────────────────────────────

    def _run_one(self, params: dict) -> dict:
        """Fit all three models with given params; return metrics dict."""
        k           = params.get("k", 5)
        n_components = params.get("n_components", 6)
        max_iter    = params.get("max_iter", 150)

        # Clip n_components to actual feature count
        n_components = min(n_components, self.dataset["n_features"])

        models = [
            KNNBaseline(k=k),
            NCAModel(k=k, n_components=n_components, max_iter=max_iter),
            LMNNModel(k=k, n_components=n_components, max_iter=max_iter),
        ]

        results = {}
        for model in models:
            name = model.name
            try:
                t0 = time.perf_counter()
                model.fit(self.X_tr, self.y_tr)
                train_time = time.perf_counter() - t0

                y_pred  = model.predict(self.X_te)
                metrics = compute_metrics(self.y_te, y_pred)
                metrics["train_time"] = round(train_time, 4)
                metrics["error"]      = None
            except Exception as exc:
                metrics = {
                    "accuracy": None, "f1_macro": None,
                    "f1_per_class": None, "confusion_matrix": None,
                    "train_time": None,
                    "error": str(exc),
                }
            results[name] = metrics

        return results

    def print_sweep(self, sweep_results: dict, param: str) -> None:
        """Print a compact table of sweep results."""
        methods = ["KNN", "NCA", "LMNN"]
        print(f"\n  Ablation: {param} sweep on '{self.dataset['name']}'")
        print(f"  {param:<10} " + "".join(f"{m:>12}" for m in methods))
        print("  " + "-" * (10 + 12 * len(methods)))
        for val, method_res in sorted(sweep_results.items()):
            row = f"  {str(val):<10} "
            for m in methods:
                acc = method_res.get(m, {}).get("accuracy")
                row += f"{acc:>12.4f}" if acc is not None else f"{'N/A':>12}"
            print(row)