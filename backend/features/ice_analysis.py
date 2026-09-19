"""Feature group: ice analysis and hazard detection.

This facade groups segmentation, detection, discrimination, tracking, and
breakup logic behind a single import surface.
"""

try:
    from backend.segmentation import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from segmentation import *

try:
    from backend.detection import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from detection import *

try:
    from backend.discriminator import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from discriminator import *

try:
    from backend.floe_filter import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from floe_filter import *

try:
    from backend.tracker import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from tracker import *

try:
    from backend.breakup_detector import *
except ModuleNotFoundError:  # pragma: no cover - legacy local execution
    from breakup_detector import *

__all__ = [
    "UNet",
    "segment_image",
    "detect_icebergs",
    "filter_ships_from_detections",
    "filter_all_detections",
    "BreakupEvent",
    "detect_breakup_events",
]
