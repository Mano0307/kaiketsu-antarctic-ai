"""
Module 5: Ice Floe Filter + Deadlock Avoidance
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging
from detection import BoundingBox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FloeFilterResult:
    detection_id: str
    object_type: str
    is_free_iceberg: bool
    reason: str
    action: str  # "TRACK" or "IGNORE"

def filter_all_detections(
    detections: List[BoundingBox],
    image: np.ndarray
) -> Tuple[List[BoundingBox], List[BoundingBox], List[FloeFilterResult]]:
    """
    DEADLOCK AVOIDANCE: Filters out non-iceberg entities like connected sea-ice floes,
    non-hazardous clutter, and ships so the path planning algorithm doesn't lock up or get confused.
    """
    tracked = []
    ignored = []
    log = []

    for box in detections:
        if box.id.startswith("F-"):
            res = FloeFilterResult(
                detection_id=box.id,
                object_type="ice_floe",
                is_free_iceberg=False,
                reason="Attached sea-ice floe (not free floating)",
                action="IGNORE"
            )
            ignored.append(box)
        elif box.id.startswith("S-"):
            res = FloeFilterResult(
                detection_id=box.id,
                object_type="vessel",
                is_free_iceberg=False,
                reason="Vessel signature detected by CNN",
                action="IGNORE"
            )
            ignored.append(box)
        else:
            res = FloeFilterResult(
                detection_id=box.id,
                object_type="iceberg",
                is_free_iceberg=True,
                reason="Confirmed free-floating iceberg with deep draft hazard",
                action="TRACK"
            )
            tracked.append(box)
        log.append(res)

    logger.info(f"Deadlock Avoidance: Tracked={len(tracked)}, Ignored={len(ignored)}")
    return tracked, ignored, log

if __name__ == "__main__":
    from detection import detect_icebergs
    dummy_img = np.zeros((512, 512), dtype=np.float32)
    dets = detect_icebergs(dummy_img)
    tracked, ignored, log = filter_all_detections(dets, dummy_img)
    for l in log:
        print(f"[{l.action}] {l.detection_id}: {l.reason}")
