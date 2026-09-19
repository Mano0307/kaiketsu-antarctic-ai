"""Feature group: forecasting and motion modeling."""

try:
    from backend.trajectory import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from trajectory import *

try:
    from backend.sea_ice_forecast import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from sea_ice_forecast import *

__all__ = [
    "predict_trajectory",
    "forecast_sea_ice",
]
