# 🧊 Kaiketsu — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory & Navigation Decision Support System

**Smart India Hackathon 2026 | Problem ID: 26059 | Team: Kaiketsu is Here**  
**Theme:** Transportation & Logistics | **PS Category:** Software

---

## 🚀 Overview

Kaiketsu is an AI-powered decision support system that helps research ships navigate safely through Antarctic waters by detecting sea ice and icebergs, predicting their movement, and computing optimized navigation routes — all in real time.

---

## 📁 Project Structure

```
kaiketsu-antarctic-ai/
│
├── frontend/                    # Claymorphism Navy Blue Web UI
│   ├── index.html               # Main HTML (13-step pipeline, map, training UI)
│   ├── styles.css               # Claymorphism Design System (Navy Blue + Ice Cyan)
│   └── app.js                   # Full interactive logic (canvas, pipeline, AI training)
│
├── backend/                     # Python AI Pipeline (feature-oriented + legacy compatibility)
│   ├── features/                # Feature packages used by the app
│   │   ├── __init__.py          # Feature registry + exports
│   │   ├── ingestion.py         # Data prep + SAR preprocessing feature
│   │   ├── ice_analysis.py      # Segmentation + detection + filtering feature
│   │   ├── forecasting.py       # Trajectory + sea-ice forecasting feature
│   │   ├── risk_and_routing.py  # Hazard scoring + route optimization feature
│   │   └── api.py               # FastAPI layer exposed as a feature
│   │
│   ├── preprocessing.py         # Module 1  — Lee Filter + SAR Calibration + Geocoding
│   ├── segmentation.py           # Module 2  — U-Net Sea-Ice Segmentation
│   ├── detection.py              # Module 3  — YOLO/ResUNet Iceberg Detection
│   ├── discriminator.py          # Module 4  — Ship-Iceberg Discriminator CNN
│   ├── floe_filter.py            # Module 5  — Ice Floe Filter + Deadlock Avoidance
│   ├── tracker.py                # Module 6  — Kalman Filter + DeepSORT Tracking
│   ├── breakup_detector.py       # Module 7  — Breakup Detection (Change Graph)
│   ├── trajectory.py             # Module 8  — Iceberg Trajectory (Physics + LSTM)
│   ├── sea_ice_forecast.py       # Module 9  — Sea-Ice ConvLSTM Forecaster
│   ├── hazard_fusion.py          # Module 10 — Hazard Risk Map (GBT Fusion)
│   ├── vessel_performance.py     # Module 11 — Vessel Performance Model (NN/GBT)
│   ├── route_optimizer.py        # Module 12 — Route Optimization (A* + NSGA-II)
│   ├── dataset_loader.py         # Dataset downloader (Kaggle/Copernicus) + DataLoaders
│   ├── main_server.py            # FastAPI REST API server
│   └── requirements.txt          # Python dependencies
│
└── README.md                     # This file
```

---

## 🔬 AI Pipeline — 13 Modules

| # | Module | Technology | Output |
|---|--------|-----------|--------|
| 01 | Preprocessing | Lee Filter + OpenCV Calibration | Clean SAR image |
| 02 | Sea-Ice Segmentation | **U-Net** (PyTorch) | Water vs ice mask |
| 03 | Iceberg Detection | **YOLO / ResUNet** (PyTorch) | Bounding boxes + confidence |
| 04 | Ship Discriminator | **CNN Classifier** (ResNet-18) | Ships removed, icebergs kept |
| 05 | Ice Floe Filter | **ML Classifier + Rules** | Deadlock avoidance — non-icebergs ignored |
| 06 | Iceberg Tracker | **Kalman Filter + DeepSORT** | Iceberg IDs across frames |
| 07 | Breakup Detector | **Change Detection + Graph** (NetworkX) | Parent → child iceberg events |
| 08 | Trajectory Model | **Physics + LSTM** (PyTorch) | 24h / 48h / 72h iceberg path |
| 09 | Sea-Ice Forecast | **ConvLSTM / U-Net Forecaster** | Future ice concentration maps |
| 10 | Hazard Fusion | **GBT Risk Scoring** (XGBoost) | Dangerous vs safe zone grid |
| 11 | Vessel Performance | **NN + GBT** (PyTorch + XGBoost) | Speed, fuel, resistance |
| 12 | Route Optimizer | **A\* + NSGA-II Pareto** | Safest, Balanced, Fastest routes |
| 13 | Dashboard/ECDIS | **FastAPI + React/JS** | REST API + Interactive UI |

---

## 🛡️ Deadlock Avoidance

The system prevents false alarms and unnecessary routing detours using a multi-stage filter:

1. **Ship Discriminator (Module 4):** Removes vessel reflections (CNN confidence < 65% → classified as ship → ignored).
2. **Ice Floe Filter (Module 5):** ML classifier + rules distinguishing free-floating icebergs from sea-ice floes attached to the ice shelf.
3. **Result:** Only confirmed free-floating icebergs with hazard potential are tracked and fed to trajectory and routing modules.

---

## 🗺️ Route Options (A\* + NSGA-II)

| Route | Risk Score | ETA | Fuel |
|-------|-----------|-----|------|
| 🛡️ **Safest** | 0.12 (Very Low) | 38h | 184 mt |
| ⚖️ **Balanced** | 0.35 (Moderate) | 31h | 160 mt |
| ⚡ **Fastest** | 0.62 (High) | 27h | 148 mt |

---

## 📦 Datasets Used

| Dataset | Source | Size |
|---------|--------|------|
| Sentinel-1 SAR Iceberg (Statoil/C-CORE) | Kaggle | ~2.5 GB |
| NSIDC Sea-Ice Concentration | NSIDC | ~1.8 GB |
| Antarctic Ice Sheet Altimetry | CryoSat-2 | ~900 MB |
| GFS Weather + Ocean Currents | Copernicus | ~400 MB |
| AIS Ship Tracking (Polar) | MarineTraffic | ~120 MB |
| Bathymetric Depth | GEBCO | ~600 MB |

---

## 🏃 Quick Start

### Frontend (Web UI)
Open `frontend/index.html` directly in your browser — no server needed!

### Backend (AI API Server)
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main_server.py

# Server running at: http://localhost:8000
# API docs at:       http://localhost:8000/docs
```

### Download Datasets (Kaggle)
```bash
# Set up Kaggle credentials first
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Then from backend/
python dataset_loader.py
```

### Test Individual Modules
```bash
python preprocessing.py
python segmentation.py
python detection.py
python route_optimizer.py
```

---

## 🔬 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze-image` | Full 13-module pipeline on uploaded SAR image |
| POST | `/api/train-model` | Train any AI module with dataset |
| GET  | `/api/training-status/{job_id}` | Check training progress |
| POST | `/api/optimize-route` | Compute 3 Pareto-optimal routes |
| GET  | `/api/download-dataset` | Download a dataset |
| GET  | `/api/health` | System health check |

---

## 📊 Impact & Benefits

- 🎯 **20–30% better** small iceberg detection (YOLO/ResUNet vs manual)
- 🌡️ **10–20% improvement** in sea-ice forecast accuracy (ConvLSTM vs single-source)
- ⛽ **3–10% fuel savings** per voyage (A* + NSGA-II route optimization)
- 🛡️ **Real-time deadlock avoidance** prevents false route deviations
- 📍 **72-hour iceberg trajectory** with confidence radius
- 🌍 **Scalable** to any polar route — no hardware modification required

---

## 👥 Team

**Kaiketsu is Here** | SIH 2026 | Problem ID: 26059 | Theme: Transportation & Logistics  
NCPOR / MoES Beneficiary — Supporting India's Antarctic Research Program

---

## 📚 References

1. ESA Sentinel-1 SAR — https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1
2. IBYOLO Iceberg Detection — https://www.sciencedirect.com/science/article/abs/pii/S0034425726001148
3. IceNet (Physics + AI Sea-Ice Forecast) — https://www.nature.com/articles/s41467-021-25257-4
4. CryoTrack Breakup Model — https://tc.copernicus.org/articles/20/467/2026/
5. Vessel Performance with NSGA-II — https://www.sciencedirect.com/science/article/pii/S0029801825037035
6. Uncertainty-Aware Polar Routing — https://www.sintef.no/en/publications/publication/019ba30d3a97-b6ec0610-1c41-49b7-861b-1c75580c499d/
