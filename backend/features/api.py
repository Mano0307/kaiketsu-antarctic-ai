"""Feature group: API and server surface."""

try:
    from backend.main_server import app
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from main_server import app

__all__ = ["app"]
