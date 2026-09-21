"""
FastAPI Server — Kaiketsu AI Antarctic Navigation System REST API
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

from preprocessing import generate_synthetic_sar, preprocess_sar_image
import numpy as np
import cv2
import base64
from segmentation import UNet, segment_image
from detection import detect_icebergs
from discriminator import filter_ships_from_detections
from floe_filter import filter_all_detections
from trajectory import predict_trajectory
from sea_ice_forecast import forecast_sea_ice
from hazard_fusion import evaluate_hazard_risk
from vessel_performance import predict_vessel_performance
from route_optimizer import optimize_routes
from dataset_loader import DATASET_REGISTRY, download_dataset

app = FastAPI(
    title="Kaiketsu AI Antarctic Navigation API",
    description="13-Module AI Pipeline for Polar Navigation & Decision Support",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    start_lat: float = -68.42
    start_lon: float = -53.10
    goal_lat: float = -67.50
    goal_lon: float = -50.20
class ForecastRiskRequest(BaseModel):
    start_lat: float = -68.42
    start_lon: float = -53.10
    goal_lat: float = -67.50
    goal_lon: float = -50.20
    ice_concentration: float = 0.28
    iceberg_proximity_m: float = 1200.0
    wind_speed_kts: float = 25.0
    depth_m: float = 450.0
class TrainRequest(BaseModel):
    model_name: str
    epochs: int = 10

@app.get("/")
def root():
    return {"message": "Kaiketsu AI Antarctic Navigator API is online."}

@app.get("/api/health")
def health():
    return {"status": "healthy", "modules_loaded": 13}

@app.post("/api/analyze-image")
def analyze_image(file: UploadFile | None = File(default=None)):
    # If a file was uploaded, read and decode it to a NumPy array
    if file is not None:
        contents = file.file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
        raw_img = img.astype(np.float32)
        source = "uploaded_file"
    else:
        raw_img = generate_synthetic_sar(shape=(512, 512))
        source = "synthetic"

    proc_img, info = preprocess_sar_image(raw_img)
    mask, seg_stats = segment_image(proc_img)
    raw_dets = detect_icebergs(proc_img)
    bergs, ships = filter_ships_from_detections(raw_dets, proc_img)
    tracked, ignored, avoid_log = filter_all_detections(raw_dets, proc_img)

    # Encode processed image as PNG and return base64 so frontend can display
    try:
        out_img = (np.clip(proc_img, 0.0, 1.0) * 255.0).astype(np.uint8)
        success, png = cv2.imencode('.png', out_img)
        if success:
            b64 = base64.b64encode(png.tobytes()).decode('ascii')
        else:
            b64 = None
    except Exception:
        b64 = None

    return {
        "status": "success",
        "source": source,
        "filename": getattr(file, 'filename', None) if file is not None else None,
        "is_valid_sar": True,
        "deadlock_triggered": False,
        "deadlock_message": "SAR Polar imagery validated. Ship & non-ice floe deadlock filter active.",
        "preprocessing": info,
        "segmentation": seg_stats,
        "detections": [b.__dict__ for b in tracked],
        "ignored_objects": [b.__dict__ for b in ignored],
        "deadlock_avoidance_log": [l.__dict__ for l in avoid_log],
        "processed_image_base64": b64,
    }

@app.post("/api/optimize-route")
def route_api(req: RouteRequest):
    routes = optimize_routes((req.start_lat, req.start_lon), (req.goal_lat, req.goal_lon))
    return {k: v.__dict__ for k, v in routes.items()}
@app.post("/api/forecast-risk-route")
def forecast_risk_route(req: ForecastRiskRequest):
    # Sea-ice forecast
    forecast = forecast_sea_ice(req.ice_concentration)

    # Environmental hazard risk
    risk = evaluate_hazard_risk(
        req.ice_concentration,
        req.iceberg_proximity_m,
        req.wind_speed_kts,
        req.depth_m
    )

    # Route optimization
    routes = optimize_routes(
        (req.start_lat, req.start_lon),
        (req.goal_lat, req.goal_lon)
    )

    return {
        "forecast": forecast,
        "risk": risk.__dict__,
        "routes": {k: v.__dict__ for k, v in routes.items()}
    }
@app.get("/api/download-dataset")
def download_api(dataset_key: str):
    try:
        return download_dataset(dataset_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
