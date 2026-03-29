# SpringHackathon-2026-VisionX
Spring Hackathon 2026 project by Team Vision X: turning parking video data into intelligent, real-time insights using AI.

Datasets: [Dragon Lake Parking](https://sites.google.com/berkeley.edu/dlp-dataset) and relative [API](https://github.com/MPC-Berkeley/dlp-dataset), [CHAD](https://github.com/TeCSAR-UNCC/CHAD?tab=readme-ov-file)

## Setup

Pull the submodule contents:
```bash
git submodule update --init
```

Create a virtual environment and install dependencies:
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./dlp-dataset
```

>[!Important]
> You have to download the data from the sources provided and put them in a `data/` folder with the following structure:
> ```
> .
> ├── processed
> └── raw
>     ├── CHAD
>     └── DLP
>         ├── json   (DLP JSON ground truth files)
>         └── raw    (DLP video files: .MOV, _data.xml)
> ```

## Pipeline Architecture

The system processes drone parking lot video through a multi-stage pipeline:

```
Video (.MOV) ──► YOLOv11 Detection ──► ByteTrack Tracking ──► Homography Mapping ──► Metrics
                  (vehicle boxes)      (persistent IDs)      (pixel → ground coords)    │
                                                                                         ▼
                                                              DLP Ground Truth ──► Evaluation
                                                                                         │
                                                                                         ▼
                                                                                    Dashboard
```

### Components

- **Detection**: YOLOv11 (pretrained on COCO) detects vehicles (car, truck, bus, motorcycle) in each frame
- **Tracking**: Ultralytics built-in ByteTrack assigns persistent track IDs across frames
- **Homography**: A pixel-to-ground-coordinate transform computed from the DLP XML annotations (which provide both pixel bounding boxes and UTM coordinates). Uses `cv2.findHomography` with RANSAC
- **Metrics**: Computed from model detections (not ground truth):
  1. **Vehicle count** — unique tracked vehicles across the scene
  2. **Real-time occupancy** — free vs occupied parking spots over time (sampled at 1s intervals)
  3. **Average dwell time** — duration vehicles remain parked in a space
  4. **Entry/exit counts** — vehicles entering/leaving through the parking lot entrance
- **Evaluation**: Compares model detections against DLP JSON ground truth (vehicle counts, spatial accuracy)
- **Dashboard**: Streamlit app with KPI cards, occupancy charts, dwell time histograms, and entry/exit timelines

### Code Structure

```
src/
  detection/
    base.py              # Abstract ParkingDetector (torch.nn.Module)
    yolo_detector.py     # YOLOv11 + ByteTrack implementation
  pipeline/
    homography.py        # Pixel ↔ ground coordinate transform
    metrics.py           # 4 metric computations
    run.py               # CLI entry point
  evaluation/
    evaluate.py          # Model vs ground truth comparison
  anomaly/
    data.py              # CHAD skeleton dataset loader
    model.py             # MPED-RNN encoder-decoder LSTM
    train.py             # Training on normal sequences
    evaluate.py          # AUC-ROC, AUC-PR, EER evaluation
dashboard/
  app.py                 # Streamlit dashboard (Vehicle Analytics + Anomaly Detection)
```

## Usage

### Run the pipeline
```bash
python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV
```

Options:
- `--model yolo11s.pt` — use a larger YOLO model for better accuracy (default: `yolo11n.pt`)
- `--conf 0.3` — detection confidence threshold (default: 0.25)
- `--imgsz 1920` — inference image size (default: 1920)
- `--sample-interval 1.0` — occupancy sampling interval in seconds

Results are saved to `data/processed/DJI_0012/`.

### Evaluate against ground truth
```bash
python -m src.evaluation.evaluate --scene DJI_0012
```

### Anomaly Detection (CHAD)

Train the skeleton-based anomaly detection model:
```bash
# Train MPED-RNN on normal skeleton sequences
python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta

# Train on specific cameras only
python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta --cameras 1 3

# Evaluate on test set
python -m src.anomaly.evaluate --model-dir models/anomaly/cam_1_2_3_4
```

### Launch dashboard
```bash
streamlit run dashboard/app.py
```

## Model Choices

- **YOLOv11n** (nano): Fastest inference, suitable for real-time. COCO pretrained weights include car/truck/bus/motorcycle classes. The drone's bird-eye view differs from COCO's typical perspective, which may reduce accuracy — fine-tuning on drone data is a future improvement.
- **ByteTrack**: Simple, fast multi-object tracker. Works well for parking lot scenarios where vehicles move slowly and occlusion is minimal from a drone view.
- **Homography**: Assumes a flat ground plane (valid for a parking lot). Computed from DLP XML annotations which provide corresponding pixel and UTM coordinates per vehicle.

## Anomaly Detection

Multi-camera anomaly detection for parking lot safety using the [CHAD](https://github.com/TeCSAR-UNCC/CHAD) dataset.

### Approach

- **Model**: MPED-RNN (Message-Passing Encoder-Decoder RNN) — an encoder-decoder LSTM with two heads:
  - **Reconstruction head**: Reconstructs the input skeleton sequence in reverse
  - **Prediction head**: Predicts future skeleton poses autoregressively
- **Training**: Unsupervised — trained only on normal skeleton sequences. High reconstruction/prediction error indicates anomalous behavior
- **Input**: Pre-extracted 17-joint COCO keypoints (x, y coordinates) from CHAD .pkl annotations — no pixel data processed
- **Privacy-by-design**: Operates exclusively on skeleton coordinates, preserving individual privacy
- **Multi-camera**: CHAD provides 4 overlapping camera views of the same parking lot; the model evaluates each camera independently

### Anomaly Types Detected

CHAD contains 22 anomalous behaviors: fighting, punching, kicking, pushing, pulling, strangling, theft, pick-pocketing, tripping, chasing, throwing, running, falling, littering, jumping, hopping, sleeping, and more.

### Metrics

- **AUC-ROC**: Area Under ROC Curve (binary classification quality)
- **AUC-PR**: Area Under Precision-Recall Curve (handles class imbalance)
- **EER**: Equal Error Rate (threshold where FPR = FNR)

CHAD paper benchmarks: MPED-RNN achieves ~0.718 AUC-ROC averaged across cameras.

## Known Limitations

- **Single scene**: Only DJI_0012 has raw video available; other scenes have JSON-only data
- **Pretrained model**: YOLOv11 is trained on COCO (street-level photos), not drone footage — detection accuracy may be lower for top-down views, especially for small or partially visible vehicles
- **Static homography**: The transform assumes the drone camera is stationary throughout the video (valid for DLP dataset)
- **Dwell time censoring**: Vehicles already parked when the video starts or still parked when it ends have truncated (censored) dwell times
- **Parking space occupancy**: Relies on mapping detection centers to ground coordinates; bounding box noise can cause false occupied/free classifications at space boundaries
- **Entrance detection**: Entry/exit classification uses a fixed entrance zone; vehicles appearing/disappearing at scene edges (not through the gate) may be misclassified
