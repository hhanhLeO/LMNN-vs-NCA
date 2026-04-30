"""
src/evaluation/metrics.py

Metric helpers used by the evaluator and ablation study.

compute_metrics(y_true, y_pred) → dict
    Returns accuracy, macro-F1, per-class F1, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics for a single prediction run.

    Returns
    -------
    dict with keys:
        accuracy        : float  — fraction of correct predictions
        f1_macro        : float  — macro-averaged F1 across all classes
        f1_per_class    : list   — per-class F1 scores
        confusion_matrix : list  — 2-D confusion matrix (list of lists)
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1c = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm  = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":         round(acc, 6),
        "f1_macro":         round(f1m, 6),
        "f1_per_class":     [round(v, 6) for v in f1c],
        "confusion_matrix": cm,
    }


def print_metrics_table(results: dict, dataset_name: str = "") -> None:
    """
    Pretty-print a results dict (method → metrics) to stdout.

    Parameters
    ----------
    results : dict
        Mapping  method_name → {"accuracy": float, "f1_macro": float, ...}
    dataset_name : str
        Optional header label.
    """
    header = f"  {dataset_name}" if dataset_name else ""
    print(header)
    print(f"  {'Method':<8} {'Accuracy':>9} {'F1 Macro':>9} {'Time (s)':>9}")
    print("  " + "-" * 38)
    for method, r in results.items():
        acc  = f"{r['accuracy']:.4f}"  if r.get("accuracy")  is not None else "  N/A  "
        f1   = f"{r['f1_macro']:.4f}"  if r.get("f1_macro")   is not None else "  N/A  "
        t    = f"{r['train_time']:.3f}" if r.get("train_time") is not None else "  N/A  "
        print(f"  {method:<8} {acc:>9} {f1:>9} {t:>9}")