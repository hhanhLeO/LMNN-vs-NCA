"""
run_all.py — Single entry point for the full experiment pipeline.

Steps
-----
1. Load config from configs/experiment_config.yaml
2. Run main evaluation on all datasets
3. Run ablation sweeps (k, n_components, max_iter)
4. Generate and save all plots
5. Save full results to results/

Usage
-----
    python run_all.py
    python run_all.py --config configs/experiment_config.yaml
    python run_all.py --skip-plots
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import yaml

from src.datasets import load_dataset
from src.models import KNNBaseline, NCAModel, LMNNModel
from src.evaluation.evaluator import Evaluator
from src.evaluation.ablation import AblationStudy
from src.visualization.plot_results import (
    plot_main_results,
    plot_ablation_sweep,
    plot_all_ablations,
)
from src.visualization.plot_embeddings import plot_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    for key in ("results_dir", "plots_dir"):
        path = cfg["output"].get(key, "results")
        os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [saved] {path}")


def _serialise(obj):
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1. Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_main_evaluation(cfg):
    print("\n" + "=" * 60)
    print("  [1/3] Main Evaluation")
    print("=" * 60)

    defaults         = cfg["defaults"]
    n_components_map = cfg.get("dataset_components", {})

    ev = Evaluator(
        test_size=defaults["test_size"],
        random_state=defaults["random_state"],
    )
    ev.set_default_models(
        k=defaults["k"],
        n_components=defaults["n_components"],
        max_iter=defaults["max_iter"],
    )

    all_ds_names = cfg["datasets"]["benchmark"] + cfg["datasets"]["novel"]
    results = ev.evaluate_all(
        dataset_names=all_ds_names,
        n_components_map=n_components_map,
        verbose=True,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ablation sweeps
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(cfg):
    print("\n" + "=" * 60)
    print("  [2/3] Ablation Study")
    print("=" * 60)

    abl_cfg  = cfg["ablation"]
    defaults = cfg["defaults"]
    ablation_results = {}

    for sweep_name, sweep_cfg in abl_cfg.items():
        ds_name = sweep_cfg["dataset"]
        param   = sweep_cfg["param"]
        values  = sweep_cfg["values"]
        fixed   = sweep_cfg.get("fixed", {})

        dataset = load_dataset(ds_name)
        study   = AblationStudy(
            dataset=dataset,
            test_size=defaults["test_size"],
            random_state=defaults["random_state"],
        )

        print(f"\n  {sweep_name}: '{param}' sweep on {dataset['name']}")
        results = study.sweep(param=param, values=values, fixed=fixed)
        study.print_sweep(results, param)

        ablation_results[sweep_name] = {
            "dataset": dataset["name"],
            "param":   param,
            "values":  values,
            "fixed":   fixed,
            "results": results,
        }

    return ablation_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plots
# ─────────────────────────────────────────────────────────────────────────────

def run_plots(cfg, main_results, ablation_results):
    print("\n" + "=" * 60)
    print("  [3/3] Generating Plots")
    print("=" * 60)

    plots_dir = cfg["output"]["plots_dir"]
    defaults  = cfg["defaults"]

    plot_main_results(
        results=main_results,
        save_path=os.path.join(plots_dir, "main_results.png"),
    )

    abl_for_combined = {}
    for sweep_name, data in ablation_results.items():
        param = data["param"]
        ds    = data["dataset"]
        res   = data["results"]
        plot_ablation_sweep(
            sweep_results=res,
            param=param,
            dataset_name=ds,
            save_path=os.path.join(plots_dir, f"ablation_{sweep_name}.png"),
        )
        abl_for_combined[sweep_name] = res

    plot_all_ablations(
        ablation_data=abl_for_combined,
        save_path=os.path.join(plots_dir, "ablation_combined.png"),
    )

    nc_map = cfg.get("dataset_components", {})
    for ds_name in ["iris", "interleaved_gaussians", "anisotropic_blobs"]:
        dataset = load_dataset(ds_name)
        nc      = min(nc_map.get(ds_name, defaults["n_components"]), dataset["n_features"])

        nca_m  = NCAModel( k=defaults["k"], n_components=nc, max_iter=defaults["max_iter"])
        lmnn_m = LMNNModel(k=defaults["k"], n_components=nc, max_iter=defaults["max_iter"])

        safe = dataset["name"].lower().replace(" ", "_")
        plot_embeddings(
            dataset=dataset,
            nca_model=nca_m,
            lmnn_model=lmnn_m,
            save_path=os.path.join(plots_dir, f"embeddings_{safe}.png"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LMNN vs NCA vs KNN — full experiment")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LMNN vs NCA vs KNN — Intro to ML Project")
    print("=" * 60)

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    main_results     = run_main_evaluation(cfg)
    save_json(_serialise(main_results),     cfg["output"]["main_results_file"])

    ablation_results = run_ablation(cfg)
    save_json(_serialise(ablation_results), cfg["output"]["ablation_results_file"])

    if not args.skip_plots:
        run_plots(cfg, main_results, ablation_results)
    else:
        print("\n  [3/3] Plots skipped (--skip-plots)")

    print("\n" + "=" * 60)
    print("  Done. Results in:", cfg["output"]["results_dir"])
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()