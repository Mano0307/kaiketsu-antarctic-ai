"""Feature group: ingestion and preprocessing.

This facade keeps the legacy module layout working while exposing a clearer
feature boundary for data ingestion and SAR preparation.
"""

try:
    from backend.dataset_loader import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from dataset_loader import *

try:
    from backend.preprocessing import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from preprocessing import *

__all__ = [
    "DATASET_REGISTRY",
    "download_dataset",
    "generate_synthetic_sar",
    "preprocess_sar_image",
]
