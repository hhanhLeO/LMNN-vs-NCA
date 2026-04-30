"""
src/visualization/plot_results.py

Bar charts for main evaluation results and line charts for ablation sweeps.

Functions
---------
plot_main_results(results, save_path)
    Grouped bar chart: accuracy for KNN / NCA / LMNN across all datasets.

plot_ablation_sweep(sweep_results, param, dataset_name, save_path)
    Line chart: accuracy vs. swept hyperparameter value.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Colour palette (accessible)
COLOURS = {
    "KNN":  "#73726c",
    "NCA":  "#3266ad",
    "LMNN": "#1D9E75",
}
METHODS = ["KNN", "NCA", "LMNN"]


# ── Shared style ─────────────────────────────────────────────────────────────

def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


# ── Main results bar chart ────────────────────────────────────────────────────

def plot_main_results(
    results: dict,
    save_path: str = "results/plots/main_results.png",
    metric: str = "accuracy",
    figsize: tuple = (13, 5),
) -> None:
    """
    Grouped bar chart: one group per dataset, one bar per method.

    Parameters
    ----------
    results   : dict  dataset_name → {method_name → metrics_dict}
    save_path : str   where to save the PNG
    metric    : str   "accuracy" or "f1_macro"
    figsize   : tuple figure dimensions in inches
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset_names = list(results.keys())
    n_ds = len(dataset_names)
    n_m  = len(METHODS)
    width = 0.22
    x     = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=figsize)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    for i, method in enumerate(METHODS):
        vals = []
        for ds in dataset_names:
            v = results[ds].get(method, {}).get(metric)
            vals.append(v if v is not None else 0.0)

        bars = ax.bar(
            x + offsets[i], vals, width,
            color=COLOURS[method], alpha=0.88,
            label=method, zorder=3,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace(" ", "\n") for name in dataset_names],
        fontsize=9,
    )
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
    ax.set_ylim(0.70, 1.05)
    ax.set_title("Test Accuracy by Dataset and Method", fontsize=12, pad=10)
    _apply_style(ax)

    patches = [mpatches.Patch(color=COLOURS[m], label=m) for m in METHODS]
    ax.legend(handles=patches, fontsize=9, frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ── Ablation line chart ───────────────────────────────────────────────────────

def plot_ablation_sweep(
    sweep_results: dict,
    param: str,
    dataset_name: str = "",
    save_path: str = "results/plots/ablation.png",
    metric: str = "accuracy",
    figsize: tuple = (7, 4),
) -> None:
    """
    Line chart: metric vs. swept hyperparameter for each method.

    Parameters
    ----------
    sweep_results : dict  param_value → {method_name → metrics_dict}
    param         : str   hyperparameter label for the x-axis
    dataset_name  : str   dataset label for the title
    save_path     : str   where to save the PNG
    metric        : str   "accuracy" or "f1_macro"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sorted_vals = sorted(sweep_results.keys())
    fig, ax = plt.subplots(figsize=figsize)

    for method in METHODS:
        ys = [
            sweep_results[v].get(method, {}).get(metric)
            for v in sorted_vals
        ]
        valid = [(x, y) for x, y in zip(sorted_vals, ys) if y is not None]
        if not valid:
            continue
        xs_v, ys_v = zip(*valid)
        ax.plot(
            xs_v, ys_v,
            marker="o", markersize=5, linewidth=2,
            color=COLOURS[method], label=method,
        )

    ax.set_xlabel(param, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
    title = f"Ablation: {param}"
    if dataset_name:
        title += f"  ({dataset_name})"
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(sorted_vals)
    ax.legend(fontsize=9, frameon=False)
    _apply_style(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ── Combined ablation figure (all three sweeps side-by-side) ─────────────────

def plot_all_ablations(
    ablation_data: dict,
    save_path: str = "results/plots/ablation_combined.png",
    metric: str = "accuracy",
) -> None:
    """
    Three-panel figure: k sweep | n_components sweep | max_iter sweep.

    Parameters
    ----------
    ablation_data : dict  sweep_name → {param_value → {method → metrics}}
    save_path     : str   output path
    metric        : str   metric to plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sweep_keys  = list(ablation_data.keys())
    n           = len(sweep_keys)
    fig, axes   = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    param_labels = {
        "k_sweep":          "k (neighbors)",
        "components_sweep": "n_components",
        "iter_sweep":       "max_iter",
    }

    for ax, key in zip(axes, sweep_keys):
        sweep_results = ablation_data[key]
        sorted_vals   = sorted(sweep_results.keys())
        param_label   = param_labels.get(key, key)

        for method in METHODS:
            ys = [
                sweep_results[v].get(method, {}).get(metric)
                for v in sorted_vals
            ]
            valid = [(x, y) for x, y in zip(sorted_vals, ys) if y is not None]
            if not valid:
                continue
            xs_v, ys_v = zip(*valid)
            ax.plot(
                xs_v, ys_v,
                marker="o", markersize=5, linewidth=2,
                color=COLOURS[method], label=method,
            )

        ax.set_xlabel(param_label, fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title() if ax == axes[0] else "", fontsize=9)
        ax.set_title(f"{param_label} sweep", fontsize=10)
        ax.set_xticks(sorted_vals)
        _apply_style(ax)

    handles = [mpatches.Patch(color=COLOURS[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Ablation Study", fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")