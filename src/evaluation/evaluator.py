"""
src/evaluation/evaluator.py

Main evaluation loop: for each dataset, runs all three models on a
stratified train/test split and records accuracy, F1, and training time.

Usage
-----
    from src.evaluation import Evaluator
    from src.datasets import load_dataset
    from src.models import KNNBaseline, NCAModel, LMNNModel

    ev = Evaluator(test_size=0.25, random_state=42)
    ev.add_model(KNNBaseline(k=5))
    ev.add_model(NCAModel(k=5, n_components=6))
    ev.add_model(LMNNModel(k=5, n_components=6))

    dataset = load_dataset("iris")
    result  = ev.evaluate_dataset(dataset)
    ev.evaluate_all(dataset_names=["iris", "wine", "digits"])
"""

import time
import traceback
import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_metrics, print_metrics_table
from src.models import KNNBaseline, NCAModel, LMNNModel
from src.datasets import load_dataset


class Evaluator:
    """
    Runs all registered models on stratified train/test splits and
    collects metrics into a structured results dict.
    """

    def __init__(self, test_size: float = 0.25, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state
        self._models: list = []

    # ── Model registration ───────────────────────────────────────────

    def add_model(self, model) -> "Evaluator":
        self._models.append(model)
        return self

    def set_default_models(
        self,
        k: int = 5,
        n_components: int = 6,
        max_iter: int = 150,
    ) -> "Evaluator":
        """Register the standard KNN / NCA / LMNN trio."""
        self._models = [
            KNNBaseline(k=k),
            NCAModel(k=k, n_components=n_components, max_iter=max_iter),
            LMNNModel(k=k, n_components=n_components, max_iter=max_iter),
        ]
        return self

    # ── Single dataset ───────────────────────────────────────────────

    def evaluate_dataset(self, dataset: dict) -> dict:
        """
        Split → fit each model → collect metrics.

        Returns
        -------
        dict  method_name → {accuracy, f1_macro, f1_per_class,
                              confusion_matrix, train_time, error, params}
        """
        X, y = dataset["X"], dataset["y"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        results = {}
        for model in self._models:
            name = model.name
            try:
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                train_time = time.perf_counter() - t0

                y_pred  = model.predict(X_te)
                metrics = compute_metrics(y_te, y_pred)
                metrics["train_time"] = round(train_time, 4)
                metrics["params"]     = model.get_params()
                metrics["error"]      = None
            except Exception as exc:
                metrics = {
                    "accuracy": None, "f1_macro": None,
                    "f1_per_class": None, "confusion_matrix": None,
                    "train_time": None,
                    "params": model.get_params(),
                    "error": traceback.format_exc(),
                }
                print(f"  [WARNING] {name} failed on '{dataset['name']}': {exc}")

            results[name] = metrics

        return results

    # ── All datasets ─────────────────────────────────────────────────

    def evaluate_all(
        self,
        dataset_names: list,
        n_components_map: dict | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate all models on each dataset.

        Parameters
        ----------
        dataset_names : list[str]
            Dataset keys to evaluate (must be registered in src/datasets).
        n_components_map : dict, optional
            Per-dataset n_components override, e.g. {"iris": 3, "digits": 8}.
        verbose : bool
            Print a table after each dataset.

        Returns
        -------
        dict  dataset_name → {method_name → metrics_dict}
        """
        n_components_map = n_components_map or {}
        all_results = {}

        for ds_name in dataset_names:
            dataset = load_dataset(ds_name)

            # Rebuild models with per-dataset n_components if given
            if ds_name in n_components_map:
                nc = n_components_map[ds_name]
                self._update_components(nc)

            if verbose:
                tag = f"[{dataset['type']}]"
                print(f"\n{tag} {dataset['name']}")

            result = self.evaluate_dataset(dataset)
            all_results[dataset["name"]] = result

            if verbose:
                print_metrics_table(result)

        return all_results

    # ── Helpers ──────────────────────────────────────────────────────

    def _update_components(self, n_components: int):
        """Update n_components on NCA and LMNN models in-place."""
        for model in self._models:
            if hasattr(model, "n_components"):
                model.n_components = n_components
                # Re-create internal sklearn/metric-learn objects with new param
                if model.name == "NCA":
                    from sklearn.neighbors import NeighborhoodComponentsAnalysis
                    model._nca = NeighborhoodComponentsAnalysis(
                        n_components=n_components,
                        max_iter=model.max_iter,
                        random_state=model.random_state,
                    )
                elif model.name == "LMNN":
                    from metric_learn import LMNN
                    model._lmnn = LMNN(
                        n_components=n_components,
                        k=model.k,
                        max_iter=model.max_iter,
                        random_state=model.random_state,
                    )
                model._fitted = False