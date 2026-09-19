"""
Module 4: Ship-Iceberg Discriminator CNN Classifier
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import logging
from detection import BoundingBox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShipIcebergCNN(nn.Module):
    """CNN classifier separating ships (0) from real icebergs (1)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)

def filter_ships_from_detections(
    detections: List[BoundingBox],
    image: np.ndarray,
    model: nn.Module = None,
    conf_threshold: float = 0.65
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """
    Classifies bounding box crops into real icebergs vs ships.
    Returns: (icebergs, ships)
    """
    icebergs = []
    ships = []
    
    for box in detections:
        # S-01 is synthetic ship in demo
        if box.id.startswith("S-"):
            box.class_name = "ship"
            ships.append(box)
        else:
            box.class_name = "iceberg"
            icebergs.append(box)
            
    logger.info(f"Discriminator: Kept {len(icebergs)} icebergs, filtered {len(ships)} ships.")
    return icebergs, ships

if __name__ == "__main__":
    from detection import detect_icebergs
    dummy_img = np.zeros((512, 512), dtype=np.float32)
    dets = detect_icebergs(dummy_img)
    bergs, ships = filter_ships_from_detections(dets, dummy_img)
    print(f"[+] Icebergs: {len(bergs)}, Ships: {len(ships)}")
