"""Feature group: risk estimation and route planning."""

try:
    from backend.hazard_fusion import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from hazard_fusion import *

try:
    from backend.vessel_performance import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from vessel_performance import *

try:
    from backend.route_optimizer import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from route_optimizer import *

__all__ = [
    "evaluate_hazard_risk",
    "predict_vessel_performance",
    "optimize_routes",
]
