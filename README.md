# VisionX — Complete Project Guide

Spring Hackathon 2026, Team VisionX: Turning parking video data into intelligent, real-time insights using AI.

---



## 1. Project Overview

VisionX is an end-to-end computer vision pipeline that processes drone (DLP) and surveillance (CHAD) video data to generate actionable intelligence.

**Core Capabilities:**
* **Parking Monitoring**: Real-time detection, tracking, and occupancy mapping.
* **Behavioral Analytics**: Dwell time analysis, entry/exit counting, and pedestrian density (Parking Stress Index).
* **Safety & Security**: Unsupervised anomaly detection using skeleton-based pose estimation.
* **Digital Twin**: 2D interactive visualization of the parking lot status.

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Detection | YOLOv11 (VisDrone-finetuned) |
| Tracking | ByteTrack (custom config for parking) |
| Anomaly Model | MPED-RNN (Encoder-Decoder LSTM) |
| Mapping | OpenCV Homography (RANSAC) |
| Dashboard | Streamlit + Plotly |

---

## 2. Architecture & Code Structure

```text
SpringHackathon-2026-VisionX/
├── src/
│   ├── detection/       # YOLOv11 & ByteTrack implementation
│   ├── pipeline/        # Homography, metrics, and CLI orchestrator
│   ├── anomaly/         # MPED-RNN (Data, Model, Train, Eval)
│   └── evaluation/      # Ground Truth comparison scripts
├── dashboard/           # Streamlit app (Live Demo & Analytics)
├── dlp-dataset/         # API for parking geometry (364 spaces)
├── data/                # Raw (DLP, CHAD) and processed datasets
└── models/              # YOLO and Anomaly RNN checkpoints
```

---

## 3. Datasets

* **Dragon Lake Parking (DLP)**: 4K Aerial Drone Video (25 FPS). Features 364 parking spaces divided into 9 areas. Used for spatial analysis, occupancy, and tracking metrics.
* **CHAD (Anomaly Detection)**: Surveillance dataset utilizing privacy-preserving skeleton annotations (17 COCO keypoints). Contains over 570k normal sequences for unsupervised training.

---

## 4. Detection, Tracking & Mapping

* **Detection & Tracking**: Utilizes YOLOv11 fine-tuned on aerial imagery. ByteTrack handles object persistence across frames.
* **Coordinate Mapping**: An OpenCV-computed 3x3 Homography matrix projects bounding box pixel coordinates into local ground coordinates (meters).
* **Parking Lot Model**: The ground coordinates are matched against predefined polygons representing 364 spaces using an optimized Point-in-Polygon lookup.

---

## 5. Metrics Computation

* **Occupancy**: Frame-by-frame calculation of occupied vs. free spaces based on homography projection.
* **Dwell Time**: Measures how long a vehicle stays parked. Incorporates a 3-second gap tolerance to prevent segment fragmentation caused by aerial detection drops.
* **Entry/Exit**: Tracks vehicles passing through a defined entrance zone coordinate boundary.
* **Parking Stress Index (PSI)**: A 4x4 spatial grid mapping the interaction intensity between pedestrians and vehicles.

---

## 6. Anomaly Detection

Built using an MPED-RNN (Message-Passing Encoder-Decoder RNN) architecture.

* **Unsupervised Learning**: The model is trained exclusively on normal sequences. It learns standard human behavior without requiring labeled anomalies.
* **Features**: 68 features per frame (34 position, 34 velocity) derived from normalized skeleton keypoints.
* **Scoring**: Anomalies are flagged based on high reconstruction (past) and prediction (future) errors.

---

## 7. Dashboard & Virtual Twin

A modular Streamlit application built for both live demonstrations and batch analytics.

* **Live Demo**: Real-time YOLO inference overlaid with tracking IDs.
* **Virtual Twin**: A Plotly-rendered 2D map updating dynamically to reflect the status (Free/Occupied) of the 364 parking spaces.
* **Analytics Tabs**: Historical occupancy charts, dwell time histograms, PSI heatmaps, and anomaly score timelines.

---

## 8. Execution Guide

### Installation
```bash
git submodule update --init
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./dlp-dataset
```

### Batch Processing Pipeline
Run the full detection and metrics pipeline on a specific video:
```bash
python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --imgsz 1024
```

### Launch Dashboard
Start the local server for the interactive interface:
```bash
streamlit run dashboard/app.py
```

### Evaluate Anomaly Model
```bash
python -m src.anomaly.evaluate --model-dir models/anomaly/checkpoint
```

---

## 9. Key Design Decisions

1. **Track Buffer (150 frames)**: The ByteTrack buffer was extended to ~6 seconds. This prevents ID reassignment for stationary parked cars during temporary occlusions or detection failures.
2. **Point-in-Polygon Accuracy**: Basing occupancy on spatial coordinates rather than tracking IDs guarantees accurate parking counts, even if the tracker experiences ID switching.
3. **Privacy-by-Design**: The anomaly detection module strictly processes skeleton joint coordinates. No raw pixel data or identifiable image features are stored or processed.
4. **Bounding Box Normalization**: Skeleton keypoints are normalized relative to their bounding boxes, allowing the anomaly model to focus entirely on human posture and motion regardless of the camera's distance.