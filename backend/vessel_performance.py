"""
Module 11: Vessel Performance Model — Speed Degradation & Fuel Consumption (NN/GBT)
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VesselPerformance:
    speed_kts: float
    fuel_rate_mt_day: float
    resistance_kn: float

class VesselPerformanceNN(nn.Module):
    def __init__(self, in_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [speed, fuel, resistance]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def predict_vessel_performance(
    base_speed_kts: float = 14.0,
    ice_concentration: float = 0.3,
    ice_thickness_m: float = 0.8
) -> VesselPerformance:
    """Predicts speed loss and increased fuel burn in ice conditions."""
    speed_loss = (ice_concentration * 4.5) + (ice_thickness_m * 1.5)
    actual_speed = max(3.0, base_speed_kts - speed_loss)
    fuel_rate = 25.0 + (ice_concentration * 20.0) + ((base_speed_kts - actual_speed) * 2.0)
    resistance = 12.0 + (ice_concentration * 15.0)

    return VesselPerformance(
        speed_kts=round(actual_speed, 2),
        fuel_rate_mt_day=round(fuel_rate, 2),
        resistance_kn=round(resistance, 2)
    )

if __name__ == "__main__":
    vp = predict_vessel_performance(14.0, 0.35, 0.6)
    print(f"[+] Performance: Speed={vp.speed_kts}kts, Fuel={vp.fuel_rate_mt_day}mt/day")
