# src/models/__init__.py
from .knn_baseline import KNNBaseline
from .nca_model import NCAModel
from .lmnn_model import LMNNModel

ALL_MODELS = {
    "KNN":  KNNBaseline,
    "NCA":  NCAModel,
    "LMNN": LMNNModel,
}