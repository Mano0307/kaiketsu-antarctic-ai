"""Feature-oriented access layer for the Kaiketsu backend.

This package groups the monolithic backend modules into domain-driven feature
areas while keeping the existing root-level files as compatibility shims.
"""

from .ingestion import *
from .ice_analysis import *
from .forecasting import *
from .risk_and_routing import *

FEATURES = {
    "ingestion": "Data preparation, SAR preprocessing, and dataset loading",
    "ice_analysis": "Segmentation, detection, filtering, tracking, and breakup analysis",
    "forecasting": "Iceberg trajectory and sea-ice prediction",
    "risk_and_routing": "Hazard scoring, vessel performance, and route optimization",
}

__all__ = [
    "FEATURES",
    "ingestion",
    "ice_analysis",
    "forecasting",
    "risk_and_routing",
]
