"""tests/test_datasets.py — Unit tests for dataset loaders."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.datasets import load_dataset, ALL_LOADERS
from src.datasets.benchmark_datasets import (
    load_iris_dataset, load_wine_dataset, load_digits_dataset,
)
from src.datasets.novel_datasets import (
    load_interleaved_gaussians, load_anisotropic_blobs, load_noisy_moons,
)

REQUIRED_KEYS = {"X", "y", "name", "description", "type",
                 "n_samples", "n_features", "n_classes",
                 "feature_names", "class_names"}


def _check_dataset(ds):
    assert REQUIRED_KEYS.issubset(ds.keys())
    assert isinstance(ds["X"], np.ndarray)
    assert isinstance(ds["y"], np.ndarray)
    assert ds["X"].shape == (ds["n_samples"], ds["n_features"])
    assert len(ds["y"]) == ds["n_samples"]
    assert len(np.unique(ds["y"])) == ds["n_classes"]
    assert ds["type"] in ("benchmark", "novel")
    assert len(ds["feature_names"]) == ds["n_features"]
    assert len(ds["class_names"])   == ds["n_classes"]


def test_iris():
    ds = load_iris_dataset()
    _check_dataset(ds)
    assert ds["n_samples"] == 150 and ds["n_features"] == 4 and ds["n_classes"] == 3

def test_wine():
    ds = load_wine_dataset()
    _check_dataset(ds)
    assert ds["n_samples"] == 178 and ds["n_features"] == 13 and ds["n_classes"] == 3

def test_digits():
    ds = load_digits_dataset()
    _check_dataset(ds)
    assert ds["n_samples"] == 1797 and ds["n_features"] == 64 and ds["n_classes"] == 10

def test_interleaved_gaussians():
    ds = load_interleaved_gaussians()
    _check_dataset(ds)
    assert ds["n_classes"] == 2 and ds["n_features"] == 14 and ds["type"] == "novel"

def test_anisotropic_blobs():
    ds = load_anisotropic_blobs()
    _check_dataset(ds)
    assert ds["n_classes"] == 3 and ds["n_features"] == 10

def test_noisy_moons():
    ds = load_noisy_moons()
    _check_dataset(ds)
    assert ds["n_classes"] == 2 and ds["n_features"] == 8

def test_all_loaders_registered():
    expected = {"iris","wine","digits","interleaved_gaussians","anisotropic_blobs","noisy_moons"}
    assert expected == set(ALL_LOADERS.keys())

def test_load_dataset_by_name():
    for name in ALL_LOADERS:
        _check_dataset(load_dataset(name))

def test_load_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nonexistent_dataset")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])