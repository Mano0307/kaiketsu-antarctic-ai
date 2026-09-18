"""
Module 3: Iceberg Detection — YOLO / ResUNet Bounding Box & Confidence Detection
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    id: str
    x: float  # Normalized [0, 1] center_x
    y: float  # Normalized [0, 1] center_y
    w: float  # Normalized [0, 1] width
    h: float  # Normalized [0, 1] height
    confidence: float
    class_name: str = "iceberg"

class ResUNetDetector(nn.Module):
    """ResUNet architecture combining residual blocks with U-Net skip connections for object detection."""
    def __init__(self, in_channels: int = 1, num_anchors: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.res1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1))
        self.down = nn.MaxPool2d(2)
        self.res2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.head = nn.Conv2d(64, num_anchors * 5, 1)  # 5: (x, y, w, h, conf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.conv_in(x)) + x.repeat(1, 32, 1, 1) if x.shape[1] == 32 else torch.relu(self.conv_in(x))
        r1 = torch.relu(self.res1(h1)) + h1
        d1 = self.down(r1)
        r2 = torch.relu(self.res2(d1))
        u1 = self.up(r2)
        out = self.head(u1)
        return out

def apply_nms(boxes: List[BoundingBox], iou_threshold: float = 0.45) -> List[BoundingBox]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    selected = []
    while boxes_sorted:
        current = boxes_sorted.pop(0)
        selected.append(current)
        boxes_sorted = [b for b in boxes_sorted if compute_iou(current, b) < iou_threshold]
    return selected

def compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    x1_min, x1_max = b1.x - b1.w/2, b1.x + b1.w/2
    y1_min, y1_max = b1.y - b1.h/2, b1.y + b1.h/2
    x2_min, x2_max = b2.x - b2.w/2, b2.x + b2.w/2
    y2_min, y2_max = b2.y - b2.h/2, b2.y + b2.h/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    b1_area = b1.w * b1.h
    b2_area = b2.w * b2.h
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def detect_icebergs(image: np.ndarray, model: nn.Module = None, conf_threshold: float = 0.5) -> List[BoundingBox]:
    # Simulated detection for demo and pipeline integrity
    boxes = [
        BoundingBox("A-01", 0.44, 0.38, 0.08, 0.08, 0.94, "iceberg"),
        BoundingBox("A-02", 0.31, 0.63, 0.05, 0.05, 0.87, "iceberg"),
        BoundingBox("A-03", 0.60, 0.29, 0.07, 0.07, 0.91, "iceberg"),
        BoundingBox("S-01", 0.43, 0.58, 0.03, 0.03, 0.82, "potential_target"),
        BoundingBox("F-01", 0.52, 0.52, 0.06, 0.06, 0.65, "potential_target")
    ]
    return [b for b in boxes if b.confidence >= conf_threshold]

if __name__ == "__main__":
    test_img = np.zeros((512, 512), dtype=np.float32)
    detections = detect_icebergs(test_img)
    print(f"[+] Detections found: {len(detections)}")
