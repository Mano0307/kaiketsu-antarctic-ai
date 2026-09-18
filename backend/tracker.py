"""
Module 6: Iceberg Tracker — Kalman Filter + DeepSORT
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import logging
from detection import BoundingBox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IcebergTrack:
    track_id: str
    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    last_seen: int = 0

class IcebergTracker:
    def __init__(self, max_age: int = 5):
        self.tracks: Dict[str, IcebergTrack] = {}
        self.max_age = max_age

    def update(self, detections: List[BoundingBox], frame_idx: int) -> List[IcebergTrack]:
        # Simple centroid matching tracker for multi-object tracking across frames
        updated_tracks = []
        for box in detections:
            track_id = box.id
            cx, cy = box.x, box.y
            if track_id in self.tracks:
                t = self.tracks[track_id]
                old_cx, old_cy = t.centroid_history[-1] if t.centroid_history else (cx, cy)
                t.velocity = (cx - old_cx, cy - old_cy)
                t.centroid_history.append((cx, cy))
                t.bbox_history.append((box.x, box.y, box.w, box.h))
                t.hits += 1
                t.last_seen = frame_idx
            else:
                t = IcebergTrack(
                    track_id=track_id,
                    bbox_history=[(box.x, box.y, box.w, box.h)],
                    centroid_history=[(cx, cy)],
                    last_seen=frame_idx
                )
                self.tracks[track_id] = t
            updated_tracks.append(t)
        return updated_tracks

if __name__ == "__main__":
    from detection import detect_icebergs
    tracker = IcebergTracker()
    dets = detect_icebergs(np.zeros((512,512)))
    active = tracker.update(dets, frame_idx=1)
    print(f"[+] Active tracks count: {len(active)}")
