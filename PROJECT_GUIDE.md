# VisionX — Complete Project Guide

Spring Hackathon 2026, Team VisionX: turning parking video data into intelligent, real-time insights using AI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Code Structure](#2-architecture--code-structure)
3. [Datasets](#3-datasets)
4. [Vehicle Detection & Tracking](#4-vehicle-detection--tracking)
5. [Homography & Coordinate Systems](#5-homography--coordinate-systems)
6. [Parking Lot Model](#6-parking-lot-model)
7. [Metrics Computation](#7-metrics-computation)
8. [Anomaly Detection](#8-anomaly-detection)
9. [Dashboard](#9-dashboard)
10. [Virtual Twin](#10-virtual-twin)
11. [Evaluation Against Ground Truth](#11-evaluation-against-ground-truth)
12. [Pipeline Execution](#12-pipeline-execution)
13. [Key Parameters & Design Decisions](#13-key-parameters--design-decisions)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

### Goal
Build an end-to-end system that takes raw drone parking lot video (DLP dataset) and surveillance video (CHAD dataset), and produces:
- Real-time vehicle detection, tracking, and classification
- Parking occupancy monitoring with a virtual twin visualization
- Dwell time analysis, entry/exit counting
- Pedestrian density and stress index mapping
- Skeleton-based anomaly detection for abnormal behavior

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Detection | YOLOv11 (VisDrone-finetuned) |
| Tracking | ByteTrack (custom config for parking) |
| Anomaly Model | MPED-RNN (encoder-decoder LSTM) |
| Coordinate Mapping | OpenCV homography (RANSAC) |
| Dashboard | Streamlit + Plotly |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |

---

## 2. Architecture & Code Structure

```
SpringHackathon-2026-VisionX/
├── src/
│   ├── detection/
│   │   ├── base.py              # Abstract detector + data classes
│   │   └── yolo_detector.py     # YOLOv11 + ByteTrack implementation
│   ├── pipeline/
│   │   ├── homography.py        # Pixel-to-ground coordinate mapping
│   │   ├── metrics.py           # All metric computations (batch)
│   │   ├── realtime.py          # Incremental metrics for live demo
│   │   └── run.py               # CLI pipeline orchestrator
│   ├── evaluation/
│   │   └── evaluate.py          # Model vs DLP ground truth comparison
│   └── anomaly/
│       ├── data.py              # CHAD dataset loader
│       ├── model.py             # MPED-RNN architecture
│       ├── train.py             # Training loop
│       ├── evaluate.py          # Test evaluation with AUC-ROC/PR/EER
│       └── realtime.py          # Frame-by-frame anomaly detector
├── dashboard/
│   ├── app.py                   # Streamlit dashboard (5 tabs)
│   └── virtual_twin.py          # Parking lot 2D map renderer
├── configs/
│   └── bytetrack_parking.yaml   # Custom ByteTrack config
├── dlp-dataset/                 # Git submodule for DLP API
│   └── dlp/
│       ├── dataset.py           # Dataset class (frames, agents, instances)
│       ├── visualizer.py        # Parking space generation
│       └── parking_map.yml      # Lot geometry definition
├── data/
│   ├── raw/
│   │   ├── DLP/
│   │   │   ├── json/            # DLP ground truth (scenes, agents, frames, instances)
│   │   │   └── raw/             # .MOV videos + XML annotations
│   │   └── CHAD/
│   │       └── CHAD_Meta/       # Skeleton .pkl, anomaly .npy, split files
│   └── processed/               # Pipeline output (JSON metrics per scene)
├── models/
│   ├── yolo11n-visdrone/        # Finetuned YOLO weights
│   └── anomaly/                 # Trained anomaly model checkpoints
└── notebooks/                   # Exploration notebooks
```

### Data Flow

```
Video (.MOV)
    │
    ├─ XML annotations ──► Homography (H matrix 3x3)
    │
    ▼
YOLOv11 + ByteTrack ──► FrameDetections (per frame: vehicles + persons with track IDs)
    │
    ├──► Vehicle Count (unique tracks, by class)
    ├──► Occupancy Timeline (vehicle centers ──H──► ground coords ──► parking space lookup)
    ├──► Dwell Times (per-track parking segments with gap tolerance)
    ├──► Entry/Exit (first/last position vs entrance zone)
    ├──► Person Count (unique person tracks)
    └──► PSI (4x4 grid density → stress index)
            │
            ▼
        JSON outputs ──► Streamlit Dashboard
```

---

## 3. Datasets

### 3.1 Dragon Lake Parking (DLP) Dataset

A drone-captured parking lot dataset from UC Berkeley.

**Video specs:**
- Format: `.MOV`, resolution ~3840x2160 (4K), ~25 fps
- Duration: ~7.5 minutes per scene (~11,300 frames)
- Aerial/top-down view of a real parking lot

**Ground truth annotations (XML):**
Each `{scene}_data.xml` file contains per-frame trajectories:
```xml
<frame id="0" timestamp="0.000000">
    <trajectory id="1" type="Car" width="1.8778" length="4.7048"
                utm_x="746975.45" utm_y="3856782.52" utm_angle="4.4702"
                speed="0.23"
                front_left_x="2475.57" front_left_y="358.40"
                front_right_x="2525.86" front_right_y="370.42"
                rear_left_x="2442.34" rear_left_y="481.77"
                rear_right_x="2492.66" rear_right_y="493.85" />
</frame>
```

Each trajectory provides:
- **Identity**: `id`, `type` (Car, Pedestrian)
- **Physical dimensions**: `width`, `length` (meters)
- **Global position**: `utm_x`, `utm_y` (UTM coordinates), `utm_angle` (radians)
- **Motion**: `speed` (m/s), `lateral_acceleration`, `tangential_acceleration`
- **Pixel bounding box**: 4 corner points (distorted + undistorted variants)

**Ground truth annotations (JSON):**
Structured as linked lists:
- `scenes`: top-level container with agent and obstacle lists
- `frames`: linked list (`next`/`prev`) with instance references
- `agents`: vehicle/pedestrian entities with type and size
- `instances`: per-frame positions with coordinates, heading, speed
- `obstacles`: static objects (parked cars at video start)

**Agent types**: Car, Bus, Truck, Pedestrian, Bicycle, Undefined

### 3.2 CHAD Dataset

Surveillance video dataset for anomaly detection with skeleton annotations.

**Skeleton annotations (`.pkl`):**
```python
{
    frame_number: {
        person_id: (
            bbox,       # [x, y, width, height] in pixels
            keypoints   # 17 COCO joints x 3 (x, y, confidence) = 51 values
        )
    }
}
```

**17 COCO keypoints**: nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle

**Anomaly labels (`.npy`):** Binary array per frame (0=normal, 1=anomalous)

**Splits:**
- `train_split_1` (unsupervised): only normal sequences for training
- `test_split_1`: all sequences with ground truth for evaluation

**4 cameras** with different viewpoints, 134 videos total.

---

## 4. Vehicle Detection & Tracking

### 4.1 Data Classes (`src/detection/base.py`)

```python
@dataclass
class DetectedVehicle:
    track_id: int                                # Persistent across frames (from ByteTrack)
    bbox: tuple[float, float, float, float]      # (x1, y1, x2, y2) in pixel coords
    confidence: float                            # Detection confidence [0, 1]
    class_name: str                              # "car", "medium vehicle", or "bus"
    center_px: tuple[float, float]               # Bounding box center (cx, cy)

@dataclass
class FrameDetections:
    frame_idx: int                               # Video frame number (0-indexed)
    timestamp: float                             # Seconds from video start (frame_idx / fps)
    vehicles: list[DetectedVehicle]              # Vehicle detections
    persons: list[DetectedVehicle]               # Person detections (reuses same dataclass)
```

`ParkingDetector` is an abstract base class (inherits `torch.nn.Module`) with one abstract method: `detect_and_track(video_path) -> list[FrameDetections]`.

### 4.2 YOLOv11 Detector (`src/detection/yolo_detector.py`)

**Model**: YOLOv11n finetuned on VisDrone (aerial/drone imagery)

**Class mapping (auto-detected based on model):**

| VisDrone Class ID | Label | DLP Equivalent |
|-------------------|-------|----------------|
| 3 | car | car |
| 4 | medium vehicle | medium vehicle |
| 8 | bus | bus |
| 0 | pedestrian | — |
| 1 | people | — |

Truck (class 5) is excluded to match DLP ground truth taxonomy.

Auto-detection logic: if `model.names[3] == "car"` and `model.names[4] in ("van", "medium vehicle")`, use VisDrone classes; otherwise fall back to COCO classes.

**`detect_and_track()` parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracker` | `configs/bytetrack_parking.yaml` | ByteTrack config path |
| `imgsz` | 1920 | Inference image width |
| `stream` | True | Memory-efficient streaming |
| `max_frames` | None | Stop after N frames |
| `conf_threshold` | 0.25 | Minimum detection confidence |

**Per-frame processing:**
1. YOLO runs detection + ByteTrack tracking (`model.track(persist=True)`)
2. For each detected box: extract class ID, filter by allowed classes, get track ID
3. Compute center: `cx = (x1+x2)/2, cy = (y1+y2)/2`
4. Classify as vehicle or person based on class ID
5. Package into `FrameDetections`

### 4.3 ByteTrack Configuration (`configs/bytetrack_parking.yaml`)

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25    # Confidence threshold for confirmed tracks
track_low_thresh: 0.1      # Threshold for low-confidence second-stage matching
new_track_thresh: 0.25     # Minimum confidence to start a new track
track_buffer: 150          # Frames to keep lost tracks alive (~6s at 25fps)
match_thresh: 0.8          # IoU similarity threshold for track-detection matching
fuse_score: True           # Fuse detection confidence with motion/IoU score
```

**Why `track_buffer: 150`?** The default (30 frames = ~1.2s) caused massive track ID reassignment for parked cars. When a stationary car's detection intermittently drops (common in aerial footage), 30 frames isn't enough. 150 frames (~6 seconds) ensures parked cars maintain their identity through brief detection gaps, reducing the unique vehicle count from ~14,000 to a realistic ~300.

---

## 5. Homography & Coordinate Systems

### 5.1 Three Coordinate Systems

| System | Scale | Origin | Usage |
|--------|-------|--------|-------|
| **Pixel** | Video resolution (~3840x2160) | Top-left corner | Detection bounding boxes |
| **UTM** | WGS84 projection (meters) | Global | XML annotations (raw) |
| **Local ground** | Meters, 140x80m lot | UTM origin point | Parking spaces, all metrics |

### 5.2 UTM-to-Local Conversion

```
UTM_ORIGIN = {"x": 747064, "y": 3856846}

local_x = UTM_ORIGIN["x"] - utm_x    (= 747064 - utm_x)
local_y = UTM_ORIGIN["y"] - utm_y    (= 3856846 - utm_y)
```

Source: `dlp-dataset/raw-data-processing/generate_tokens.py`

### 5.3 Homography Computation (`src/pipeline/homography.py`)

**Goal**: Find a 3x3 matrix H that maps pixel coordinates to local ground coordinates.

**Step 1 — Collect correspondences from XML:**
```python
parse_xml_correspondences(xml_path, max_frames=50, stride=10)
```
- Sample frames at stride 10 (up to 50 frames) for spatial diversity
- For each vehicle trajectory in sampled frames:
  - Pixel center = average of 4 bbox corners (front_left, front_right, rear_left, rear_right)
  - Ground point = UTM coords converted to local
- Returns `(N, 2)` pixel points and `(N, 2)` ground points

**Step 2 — Compute homography:**
```python
H, mask = cv2.findHomography(pixel_pts, ground_pts, cv2.RANSAC, 5.0)
```
- RANSAC with reprojection threshold = 5.0 pixels
- Reports inlier ratio for quality assessment

**Step 3 — Transform points:**
```python
def pixel_to_ground(H, px, py) -> (gx, gy):
    pt = H @ [px, py, 1]     # Homogeneous multiplication
    pt /= pt[2]              # Normalize by Z
    return (pt[0], pt[1])    # Local ground coords
```

Batch version `pixels_to_ground(H, pts)` for vectorized transforms.

**Persistence**: Saved as `homography.npy` (3x3 float64 matrix).

---

## 6. Parking Lot Model

### 6.1 Parking Map (`dlp-dataset/dlp/parking_map.yml`)

**Lot dimensions**: 140m x 80m (`MAP_SIZE: {x: 140, y: 80}`)

**9 parking areas (A-I), 364 total spaces:**

| Area | Position | Grid | Spaces |
|------|----------|------|--------|
| A | Top row | 1x42 | 42 |
| B | Left, row 1 | 2x25 | 50 |
| C | Right, row 1 | 2x21 | 42 |
| D | Left, row 2 | 2x25 | 50 |
| E | Right, row 2 | 2x21 | 42 |
| F | Left, row 3 | 2x25 | 50 |
| G | Right, row 3 | 2x21 | 42 |
| H | Left, row 4 | 1x25 | 25 |
| I | Right, row 4 | 1x21 | 21 |

**Entrance zone**: approximately [5, 70] to [25, 80] in local coords (upper-left area).

**Waypoints**: 8 row-level routes, 2 column routes, 8 intersections, entrance/exit zones — used for navigation reference.

### 6.2 Parking Space Generation (`dlp-dataset/dlp/visualizer.py`)

The `Visualizer` class generates individual parking spaces from area bounds using bilinear interpolation:

1. Each area has 4-corner bounds and a grid shape (rows x cols)
2. `_divide_rect()` interpolates corner positions:
   ```
   For each row i in [0, rows]:
       left_edge[i]  = lerp(top_left, bottom_left, i/rows)
       right_edge[i] = lerp(top_right, bottom_right, i/rows)
   For each cell (r, c):
       corners = interpolated quadrilateral from edges
   ```
3. Result: `parking_spaces` DataFrame with 364 rows:
   - Columns: `id, area, top_left_x, top_left_y, top_right_x, top_right_y, btm_right_x, btm_right_y, btm_left_x, btm_left_y`
   - All coordinates in local ground meters

Spaces are not necessarily perfect rectangles — bilinear interpolation produces quadrilaterals that conform to curved area boundaries.

### 6.3 Fast Parking Space Lookup (`_ParkingSpaceLookup`)

Precomputed vectorized lookup for O(1) point-in-space queries:

```python
class _ParkingSpaceLookup:
    def __init__(self, parking_spaces):
        # Extract 4 corner x,y for each of 364 spaces
        # Precompute per-space axis-aligned bounding boxes:
        self.min_x, self.max_x = xs.min(axis=1), xs.max(axis=1)
        self.min_y, self.max_y = ys.min(axis=1), ys.max(axis=1)

    def find_space_id(self, gx, gy) -> (space_id, area):
        mask = (gx >= min_x) & (gx <= max_x) & (gy >= min_y) & (gy <= max_y)
        # Return first matching space or (None, None)
```

---

## 7. Metrics Computation

### 7.1 Vehicle Count (`compute_vehicle_count`)

Counts unique vehicles across all frames by collecting distinct `track_id` values.

**Output:**
```json
{
    "total_unique": 305,
    "by_class": {"car": 250, "medium vehicle": 50, "bus": 5},
    "per_frame_counts": [
        {"frame_idx": 0, "timestamp": 0.0, "count": 210},
        ...
    ]
}
```

### 7.2 Occupancy Timeline (`compute_occupancy_timeline`)

Measures how many parking spaces are occupied over time.

**Algorithm:**
1. Sample frames at `sample_interval` (default 1.0 second)
2. For each sampled frame:
   - Transform every vehicle's pixel center → ground coords via homography
   - Check which parking space each vehicle falls in (`_ParkingSpaceLookup`)
   - Count distinct occupied spaces (a space counts once even if multiple detections overlap)
3. Record per-area breakdown

**Output includes:**
- `timestamps`, `occupied`, `free` arrays (one value per sample)
- `total_spaces` (364)
- `by_area`: per-area occupied counts and totals
- `occupied_space_ids`: list of occupied space ID lists per timestamp (for virtual twin)

### 7.3 Dwell Times (`compute_dwell_times`)

Measures how long each vehicle stays parked.

**Algorithm:**
1. Build per-track timeline: list of `(timestamp, ground_x, ground_y)`
2. Walk through each track's timeline:
   - When vehicle enters a parking space → start timer
   - When vehicle leaves → check gap tolerance
   - If gap < 3 seconds: ignore (detection jitter), keep segment going
   - If gap >= 3 seconds: finalize segment
   - If still parked at video end: finalize with censoring flag
3. Minimum duration: 2 seconds (filter noise)

**Censoring**: A dwell event is marked `"censored": true` if it starts within 1 second of video start or ends within 1 second of video end, indicating the true duration is longer than measured.

**Output:**
```json
{
    "dwell_times": [
        {"track_id": 42, "duration_sec": 320.5, "area": "B", "censored": true}
    ],
    "stats": {
        "mean_sec": 280.0, "median_sec": 310.0,
        "min_sec": 2.5, "max_sec": 452.0,
        "count": 250, "censored_count": 180
    }
}
```

### 7.4 Entry/Exit Detection (`compute_entry_exit`)

Counts vehicles entering and exiting through the parking lot entrance.

**Entrance zone**: `x=[0, 30], y=[65, 80]` in local ground coordinates.

**Algorithm:**
- For each track, record first and last observed position (ground coords)
- **Entry**: first position falls inside entrance zone
- **Exit**: last position falls inside entrance zone
- Build cumulative timeline in 30-second bins

**Output:**
```json
{
    "entries": [{"track_id": 1, "timestamp": 5.2, "class": "car"}],
    "exits": [...],
    "entry_count": 50, "exit_count": 45,
    "timeline": {
        "timestamps": [0, 30, 60, ...],
        "cumulative_entries": [2, 8, 15, ...],
        "cumulative_exits": [0, 3, 10, ...]
    }
}
```

### 7.5 Person Count (`compute_person_count`)

Same logic as vehicle count but using `frame.persons`.

### 7.6 Parking Stress Index (`compute_psi`)

Measures pedestrian-vehicle interaction stress across spatial zones.

**Grid**: Divide video frame into `cols x rows` cells (default 4x4 = 16 zones).

**Per-zone per-frame**: Count pedestrians and vehicles whose center falls in that cell.

**PSI formula:**
```
PSI = (0.4 * norm(avg_pedestrians) + 0.4 * norm(avg_vehicles) + 0.2 * norm(avg_ratio)) * 10
```

Where:
- `norm(x) = (x - min) / (max - min)` across all zones (min-max normalization)
- `ratio = pedestrians / max(1, vehicles)` per zone
- Scale: 0 to 10 (0 = no stress, 10 = maximum stress)

**Peak PSI**: Computed per-frame then taking the max, using frame-level normalization.

### 7.7 Real-Time Metrics (`MetricsAccumulator`)

Incremental version of all metrics for the live demo. Maintains state and updates frame-by-frame.

**Key differences from batch:**
- Tracks are finalized when "lost" (not seen for 30 frames)
- `get_snapshot()` computes **active dwell times** on the fly — walks back through each currently-parked vehicle's timeline to find when it first entered a space
- Stats combine both completed and active dwells (so cars parked the entire video are counted)
- Exposes `current_occupied_space_ids` for the virtual twin

---

## 8. Anomaly Detection

### 8.1 Overview

Unsupervised skeleton-based anomaly detection using the CHAD dataset. The model learns what "normal" human behavior looks like from skeleton sequences, then flags deviations.

**Key insight**: By operating on skeleton keypoints only (not raw pixels), the system is privacy-preserving — no identifiable image data is processed.

### 8.2 Feature Engineering (`src/anomaly/data.py`)

**Input dimensionality**: 68 features per frame:
- 34 position features: 17 keypoints x 2 (x, y), confidence dropped
- 34 velocity features: frame-to-frame differences of position features

**Normalization**: Keypoint coordinates are normalized relative to bounding box:
```python
xy = (xy - bbox_center) / bbox_size
```
This makes the model invariant to absolute image position — it only sees body proportions and motion.

**Sequence construction:**
| Parameter | Training | Testing |
|-----------|----------|---------|
| `seq_len` | 12 frames | 12 frames |
| `pred_len` | 6 frames | 6 frames |
| `stride` | 2 | 1 |
| `max_gap` | 5 frames | 5 frames |

- `stride`: step between consecutive windows (2 for training = skip every other window; 1 for testing = evaluate every window)
- `max_gap`: if temporal gap > 5 frames within a person track, split into separate segments (avoids learning non-contiguous motion)

**Training data filtering**: Anomalous sequences are automatically excluded for unsupervised learning — the model only sees normal behavior.

### 8.3 Model Architecture (`src/anomaly/model.py`)

**MPED-RNN** (Message-Passing Encoder-Decoder RNN), based on Morais et al., CVPR 2019.

```
Input sequence (12 frames x 68 features)
            │
            ▼
    ┌───────────────┐
    │   Encoder     │  LSTM: input=68, hidden=128, layers=2, dropout=0.3
    │   (12 steps)  │
    └───────┬───────┘
            │ hidden state (h, c)
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
┌──────────┐   ┌──────────┐
│Recon     │   │Predict   │  Both: LSTM hidden=128, layers=2, dropout=0.3
│Decoder   │   │Decoder   │
│(12 steps)│   │(6 steps) │
└────┬─────┘   └────┬─────┘
     │ FC 128→68    │ FC 128→68
     ▼               ▼
Reconstructed    Predicted
Input (12 fr)    Future (6 fr)
```

**Encoder**: Processes the 12-frame input sequence, outputs final hidden state `(h, c)`.

**Reconstruction Decoder**: Takes reversed input sequence (teacher forcing) + encoder hidden state, reconstructs the input in original frame order.

**Prediction Decoder**: Takes last frame of input + encoder hidden state, autoregressively generates 6 future frames (each prediction feeds as the next input).

**Total parameters**: ~500K

### 8.4 Training (`src/anomaly/train.py`)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch size | 256 |
| Epochs | 80 |
| Gradient clipping | max norm 1.0 |
| Reconstruction weight | 0.5 |
| Training sequences | 576,754 (normal only, 4 cameras) |

**Loss function:**
```
total_loss = 0.5 * MSE(reconstruction, input) + 0.5 * MSE(prediction, target)
```

**Training progression:**
- Epoch 1: loss = 0.9754
- Epoch 80: loss = 0.0418 (96% reduction)
- Final reconstruction loss: 0.0207
- Final prediction loss: 0.0630

**Checkpointing**: Best model saved on lowest validation loss (`best_model.pt`), plus final model (`final_model.pt`).

### 8.5 Anomaly Scoring

```
recon_error = mean((reconstructed_input - actual_input)^2)
pred_error  = mean((predicted_future - actual_future)^2)
anomaly_score = 0.5 * recon_error + 0.5 * pred_error
```

Higher scores = more deviation from learned normal patterns = more anomalous.

### 8.6 Evaluation (`src/anomaly/evaluate.py`)

**Test set**: 237,420 sequences (57.7% anomalous, 42.3% normal)

**Overall metrics:**
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.563 |
| AUC-PR | 0.626 |
| EER | 0.452 |
| EER threshold | 0.341 |

**Per-camera breakdown:**
| Camera | AUC-ROC | AUC-PR | EER | Sequences |
|--------|---------|--------|-----|-----------|
| 1 | 0.603 | 0.705 | 0.422 | 53,164 |
| 2 | 0.505 | 0.518 | 0.494 | 48,451 |
| 3 | 0.538 | 0.533 | 0.470 | 76,035 |
| 4 | 0.597 | 0.727 | 0.425 | 59,770 |

Cameras 1 and 4 perform best (AUC-PR > 0.70). Camera 2 is near random.

### 8.7 Real-Time Detector (`src/anomaly/realtime.py`)

Frame-by-frame anomaly detection for the live demo.

**Processing pipeline per frame:**
1. Extract skeleton keypoints for each person from pre-loaded `.pkl` annotations
2. Append to per-person sliding window buffer (max 18 frames = seq_len + pred_len)
3. If temporal gap > 5 frames: reset buffer (new track segment)
4. When buffer reaches 18 frames:
   - Normalize positions relative to bounding box
   - Compute velocity features (frame-to-frame diffs)
   - Stack into batch `(N_persons x 18 x 68)`
   - Split into input (12 frames) + target (6 frames)
   - Forward through MPED-RNN → per-person anomaly scores
5. Return `FrameAnomalyResult`:
   - `scores`: per-person anomaly scores
   - `max_score`, `mean_score`: aggregated
   - `is_anomaly`: `max_score > threshold`
   - `gt_label`: ground truth (0 or 1)

**Alert logic**: An alert is counted only on the **rising edge** — when the score first crosses above threshold. Consecutive above-threshold frames count as one alert. A new alert triggers only after the score drops below threshold and rises again.

---

## 9. Dashboard

### Overview

Streamlit app with 5 tabs (`dashboard/app.py`). Uses Plotly for all interactive charts.

### Tab 1: Live Demo

Real-time YOLO inference on DLP video with live metrics and virtual twin.

**Controls:**
- Video selector (`.MOV` files from `data/raw/DLP/raw/`)
- Frame skip slider (1-15, default 5)
- Video length % slider (10-100%, default 40%) — limits how much of the video to process
- Model path input

**Layout (3:2 columns):**

Left column:
- Annotated video frame (0.5x scale, green bounding boxes with `#trackID className confidence`)
- FPS counter, progress bar
- Virtual twin parking map (updates every 10 processed frames)

Right column:
- KPI cards: Unique Vehicles, In Frame, Occupancy (X/364, Y%), Entries/Exits
- Occupancy over time chart (tomato fill)
- Dwell time histogram (20 bins, steelblue)
- Completed/active parked count + average duration

**Inference config**: `imgsz=1280`, `conf=0.25`, custom ByteTrack tracker, vehicle classes only.

### Tab 2: Vehicle Analytics

Batch results visualization from pipeline output.

**Data sources**: `vehicle_count.json`, `occupancy_timeline.json`, `dwell_times.json`, `entry_exit.json`, plus optional `baseline_occupancy.json`, `gt_occupancy.json`, `evaluation.json`.

**Visualizations:**
- KPI row: total vehicles + class breakdown, avg occupancy %, avg dwell time, entries/exits
- Occupancy over time with toggles for: per-area breakdown, frame-diff baseline overlay (gray dashed), ground truth overlay (gold solid)
- Dwell time histogram + per-area summary table
- Cumulative entries (dodgerblue) and exits (coral) timeline
- Per-frame detection count (downsampled if >500 points)
- Model evaluation metrics vs ground truth (if available)

### Tab 3: Pedestrian Analytics

Person counting and Parking Stress Index visualization.

**KPIs**: Total unique persons, avg/peak persons per frame, peak PSI zone.

**Visualizations:**
- Persons over time (line chart)
- PSI heatmap (4x4 grid, YlOrRd colorscale, 0-10 scale)
- Zone details table sorted by avg PSI (columns: Zone, Avg PSI, Peak PSI, Avg Peds, Avg Vehicles)

### Tab 4: Anomaly Demo

Real-time skeleton-based anomaly detection on CHAD video.

**Controls**: Model selector, video selector, frame skip (1-5), threshold (default 2.0).

**Display:**
- Video frame with overlay: red border + "ANOMALY DETECTED" when anomalous, orange border when GT anomaly but model missed
- Score and threshold text overlay, person count
- KPI cards: current score, persons in frame, alerts triggered, ground truth label
- Score timeline chart with red dashed threshold line and ground truth markers
- Alert panel with error message on anomaly detection

### Tab 5: Anomaly Detection

Batch evaluation results for trained anomaly models.

**Sections:**
- KPIs: AUC-ROC, AUC-PR, EER, test sequence count
- Per-camera performance with status indicators (green >= 0.7, orange >= 0.6, red < 0.6 AUC-ROC)
- Camera comparison bar chart
- Per-video anomaly score timeline (selectable, sorted by max score)
- Alert table (top 20 videos above threshold)
- Model configuration expander (full training config + history)
- Privacy-by-design notice

---

## 10. Virtual Twin

### Rendering (`dashboard/virtual_twin.py`)

`render_parking_map(parking_spaces_df, occupied_space_ids, title="")` returns a Plotly figure.

**How it works:**
1. For each of 364 parking spaces, draw a filled quadrilateral using Plotly `add_shape(type="path")`
2. **Occupied**: crimson fill `rgba(220, 20, 60, 0.7)`, dark red border
3. **Free**: forest green fill `rgba(34, 139, 34, 0.6)`, dark green border
4. Invisible scatter traces at space centers provide hover tooltips: "Space {id} (Area {area}) — Status: {Free/Occupied}"
5. Entrance zone drawn as dashed blue rectangle at [5,70]-[25,80] with "Entrance" label
6. Summary annotation: "Occupied: X/364 (Y%)" with dark semi-transparent background
7. Dark theme: `rgb(40,40,40)` plot background, white text, hidden axes
8. Equal aspect ratio via `yaxis.scaleanchor="x"`

**Integration:**
- Live Demo tab: renders below video frame, updates every 10 processed frames using `MetricsAccumulator.get_snapshot()["occupancy"]["occupied_space_ids"]`

---

## 11. Evaluation Against Ground Truth

### Vehicle Count Evaluation (`src/evaluation/evaluate.py`)

**Ground truth counting:**
- Moving vehicles: agents where `type not in {"Pedestrian", "Undefined", "Bicycle"}`
- Static vehicles: obstacles in scene (parked cars already present at video start)
- Total GT = moving + static

**Per-frame count evaluation:**
- Sample every 25th frame
- Compare detected vehicle count vs GT count per frame
- Metrics: MAE, median AE, max AE

**YOLO validation** (if dataset YAML available):
- Runs `model.val()` with DLP evaluation classes [3, 4, 8]
- Reports mAP50, mAP50-95, precision, recall, per-class breakdown

### Occupancy Evaluation

Baseline (frame-diff) and ground truth occupancy timelines can be overlaid on the model's occupancy chart in the Vehicle Analytics tab for visual comparison.

---

## 12. Pipeline Execution

### Batch Pipeline (`python -m src.pipeline.run`)

```bash
# Full pipeline (detection + metrics)
python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV

# Skip detection, recompute metrics only
python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --skip-detection

# Process only first 4500 frames (~40% of video)
python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --max-frames 4500
```

**All CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Path to video file |
| `--model` | `models/yolo11n-visdrone/weights/bestVisDrone.pt` | YOLO model path |
| `--conf` | 0.25 | Detection confidence threshold |
| `--imgsz` | 1920 | Inference image size (width) |
| `--sample-interval` | 1.0 | Occupancy sampling interval (seconds) |
| `--grid` | 4 4 | PSI grid dimensions (cols rows) |
| `--skip-detection` | false | Reuse cached `detections.json` |
| `--max-frames` | None | Stop after N frames |

**Pipeline stages:**
1. **Homography**: Load from cache or compute from XML → `homography.npy`
2. **Detection**: Run YOLO + ByteTrack or load cached → `detections.json`
3. **Vehicle count** → `vehicle_count.json`
4. **Spatial metrics** (if homography available):
   - Load 364 parking spaces from DLP Visualizer
   - Occupancy timeline → `occupancy_timeline.json`
   - Dwell times → `dwell_times.json`
   - Entry/exit → `entry_exit.json`
5. **Pedestrian metrics**:
   - Person count → `person_count.json`
   - PSI → `psi.json`

### Anomaly Pipeline

```bash
# Train on CHAD (normal sequences only)
python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta

# Evaluate trained model
python -m src.anomaly.evaluate --model-dir models/anomaly/cam_1_2_3_4

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 13. Key Parameters & Design Decisions

### Parameter Summary

| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| YOLO | `conf_threshold` | 0.25 | Balance precision/recall for aerial vehicle detection |
| YOLO (batch) | `imgsz` | 1920 | Native 4K width, best accuracy |
| YOLO (live) | `imgsz` | 1280 | Faster inference for real-time, acceptable accuracy trade-off |
| ByteTrack | `track_buffer` | 150 frames (~6s) | Prevent track ID reassignment for stationary parked cars |
| Homography | RANSAC threshold | 5.0 px | Robust to annotation noise in XML correspondences |
| Homography | `max_frames` / `stride` | 50 / 10 | Sample diversity without overloading RANSAC |
| Occupancy | `sample_interval` | 1.0s | 1 Hz temporal resolution, balances precision vs computation |
| Dwell time | `gap_tolerance` | 3.0s | Tolerate detection jitter without fragmenting dwell segments |
| Dwell time | `min_duration` | 2.0s | Filter spurious very-short "parked" detections |
| Entry/exit | entrance zone | [0-30, 65-80] m | Derived from parking_map.yml WAYPOINTS.EXT |
| Entry/exit | timeline bins | 30s | Coarse-grained cumulative visualization |
| PSI | grid | 4x4 (16 zones) | Meaningful spatial granularity without over-segmenting |
| PSI | weights | 0.4 ped + 0.4 veh + 0.2 ratio | Equal pedestrian/vehicle weight, ratio as tiebreaker |
| MPED-RNN | seq_len / pred_len | 12 / 6 frames | 18-frame window for temporal context |
| MPED-RNN | hidden_dim / layers | 128 / 2 | Balance capacity vs training speed (~500K params) |
| MPED-RNN | recon_weight | 0.5 | Equal balance of reconstruction and prediction signals |
| MPED-RNN | dropout | 0.3 | Regularization between LSTM layers |
| Training | batch_size / epochs | 256 / 80 | Stable convergence with large dataset |
| Training | LR schedule | ReduceLROnPlateau(0.5, patience=5) | Adaptive learning rate decay |
| Anomaly | `max_gap` | 5 frames | Prevent learning from non-contiguous skeleton tracks |
| Live demo | frame skip | 5 (default) | Balance processing speed vs temporal resolution |
| Live demo | video length | 40% (default) | Keep demo under ~3 minutes |
| Live demo | twin update | every 10 frames | Reduce Plotly rendering overhead |

### Architectural Decisions

1. **Unsupervised anomaly training**: Model only sees normal sequences. Anomalies are detected as high reconstruction/prediction error — no labeled anomaly data needed for training.

2. **Dual-head anomaly model**: Reconstruction + prediction provide complementary signals. A normal sequence that reconstructs well might still fail to predict future motion, and vice versa.

3. **Skeleton-only processing**: Privacy-preserving — no pixel-level image data stored or transmitted. Only body joint coordinates are processed.

4. **Homography-based spatial reasoning**: Instead of trying to detect parking space occupancy in pixel space, project detections to a calibrated ground plane and match against known space polygons. More robust to camera angle and distortion.

5. **Bbox normalization for skeletons**: Makes the anomaly model invariant to absolute image position — focuses on body proportions and motion patterns regardless of where in the frame the person appears.

6. **Gap-tolerant dwell time**: Detection of parked cars in aerial footage is inherently noisy. Without gap tolerance, a single frame's detection failure would fragment a 5-minute dwell into many sub-minute segments.

7. **DLP class alignment**: Using only car/medium vehicle/bus (dropping truck) ensures consistency between detected classes and ground truth evaluation classes.

---

## 14. Dependencies

### Core
- `torch==2.11.0` — PyTorch deep learning framework
- `ultralytics==8.4.30` — YOLOv11 detection + ByteTrack tracking
- `opencv-python==4.13.0.92` — Video I/O, image processing, homography
- `streamlit==1.55.0` — Dashboard web framework
- `plotly==6.6.0` — Interactive visualization

### Data Science
- `numpy==2.4.3` — Array operations
- `pandas==2.3.3` — DataFrames (parking spaces, metrics)
- `scipy==1.17.1` — Scientific computing (AUC, interpolation)
- `scikit-learn==1.8.0` — Metrics (ROC, PR curves)

### Supporting
- `pyyaml==6.0.3` — Parking map YAML parsing
- `tqdm==4.67.3` — Progress bars
- `lap==0.5.13` — Linear assignment for ByteTrack
- `pillow==12.1.1` — Image operations
- `matplotlib==3.10.8` — Static plots (notebooks)

### Setup
```bash
git submodule update --init
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./dlp-dataset
```
