# src/datasets/__init__.py
from .benchmark_datasets import load_iris_dataset, load_wine_dataset, load_digits_dataset
from .novel_datasets import (
    load_interleaved_gaussians,
    load_anisotropic_blobs,
    load_noisy_moons,
)

BENCHMARK_LOADERS = {
    "iris":   load_iris_dataset,
    "wine":   load_wine_dataset,
    "digits": load_digits_dataset,
}

NOVEL_LOADERS = {
    "interleaved_gaussians": load_interleaved_gaussians,
    "anisotropic_blobs":     load_anisotropic_blobs,
    "noisy_moons":           load_noisy_moons,
}

ALL_LOADERS = {**BENCHMARK_LOADERS, **NOVEL_LOADERS}


def load_dataset(name: str) -> dict:
    """Load any dataset by name. Returns the standard dataset dict."""
    if name not in ALL_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(ALL_LOADERS)}")
    return ALL_LOADERS[name]()