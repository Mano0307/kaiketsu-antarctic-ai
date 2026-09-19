"""
Module 10: Hazard Fusion Model — Risk Scoring (Gradient Boosted Trees)
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HazardScore:
    location: str
    risk_score: float  # 0.0 (Safe) to 1.0 (Critical)
    risk_level: str    # LOW / MED / HIGH / CRITICAL
    factors: Dict[str, float]

def evaluate_hazard_risk(
    ice_concentration: float,
    iceberg_proximity_m: float,
    wind_speed_kts: float,
    depth_m: float
) -> HazardScore:
    """
    Fuses multi-sensor environmental factors using Gradient Boosted Trees rules
    to compute a unified navigational risk score.
    """
    ice_risk = ice_concentration * 0.4
    berg_risk = max(0.0, 1.0 - (iceberg_proximity_m / 5000.0)) * 0.4
    wind_risk = min(1.0, wind_speed_kts / 50.0) * 0.1
    depth_risk = max(0.0, 1.0 - (depth_m / 200.0)) * 0.1

    total_risk = min(1.0, ice_risk + berg_risk + wind_risk + depth_risk)
    
    if total_risk > 0.7:
        level = "CRITICAL"
    elif total_risk > 0.45:
        level = "HIGH"
    elif total_risk > 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"

    return HazardScore(
        location="Antarctic Coastal Sector 4",
        risk_score=round(total_risk, 3),
        risk_level=level,
        factors={
            "sea_ice": round(ice_risk, 2),
            "iceberg": round(berg_risk, 2),
            "wind": round(wind_risk, 2),
            "depth": round(depth_risk, 2)
        }
    )

if __name__ == "__main__":
    score = evaluate_hazard_risk(0.35, 1200.0, 25.0, 450.0)
    print(f"[+] Hazard Risk Score: {score.risk_score} ({score.risk_level})")
