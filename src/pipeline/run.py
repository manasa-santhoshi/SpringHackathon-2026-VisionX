"""
CLI entry point for the parking analytics pipeline.

Usage:
    python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV
    python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --model yolo11s.pt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for DLP submodule
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))

from src.detection.yolo_detector import YOLODetector
from src.pipeline.run_congestion import CongestionAnalyzer
from src.pipeline.homography import compute_homography, load_homography, save_homography
from src.pipeline.metrics import (
    compute_dwell_times,
    compute_entry_exit,
    compute_occupancy_timeline,
    compute_vehicle_count,
)

# DLP Dataset imports (required for get_parking_spaces)
from dlp.dataset import Dataset
from dlp.visualizer import Visualizer

def get_parking_spaces() -> "pd.DataFrame":
    """Load parking spaces DataFrame from the DLP Visualizer."""
    ds = Dataset()
    # Load any scene just to initialize the Visualizer (parking spaces are scene-independent)
    data_dir = PROJECT_ROOT / "data" / "raw" / "DLP" / "json"
    scene_files = sorted(data_dir.glob("*_scene.json"))
    if not scene_files:
        raise FileNotFoundError(f"No scene JSON files found in {data_dir}")
    scene_name = scene_files[0].stem.replace("_scene", "")
    ds.load(str(data_dir / scene_name))
    vis = Visualizer(ds)
    return vis.parking_spaces


def get_scene_name(video_path: str) -> str:
    """Extract scene name from video path (e.g., 'DJI_0012' from '...DJI_0012.MOV')."""
    return Path(video_path).stem


def main():
    parser = argparse.ArgumentParser(description="Parking lot analytics pipeline")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models/yolo11n-dlp/weights/best.pt"),
        help="YOLO model path (default: DLP-finetuned model)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1920,
        help="Inference image size",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Occupancy sampling interval in seconds",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    scene_name = get_scene_name(args.video)

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "processed" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Homography
    # ─────────────────────────────────────────────────────────────────────────
    homography_path = output_dir / "homography.npy"
    xml_path = video_path.parent / f"{scene_name}_data.xml"

    if homography_path.exists():
        print(f"Loading cached homography from {homography_path}")
        H = load_homography(str(homography_path))
    elif xml_path.exists():
        print(f"Computing homography from {xml_path}")
        H = compute_homography(str(xml_path))
        save_homography(H, str(homography_path))
    else:
        print(f"WARNING: No XML annotation found at {xml_path}")
        print("Cannot compute homography. Metrics requiring ground coords will be unavailable.")
        H = None

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Detection + Tracking
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nRunning YOLOv11 detection + tracking on {video_path}")
    detector = YOLODetector(model_name=args.model, conf_threshold=args.conf)
    detections = detector.detect_and_track(str(video_path), imgsz=args.imgsz)
    print(f"Processed {len(detections)} frames")

    # Save raw detections for reuse
    detections_data = []
    for frame in detections:
        detections_data.append({
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "vehicles": [
                {
                    "track_id": v.track_id,
                    "bbox": list(v.bbox),
                    "confidence": v.confidence,
                    "class_name": v.class_name,
                    "center_px": list(v.center_px),
                }
                for v in frame.vehicles
            ],
        })
    with open(output_dir / "detections.json", "w") as f:
        json.dump(detections_data, f)
    print(f"Saved detections to {output_dir / 'detections.json'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Compute Metrics
    # ─────────────────────────────────────────────────────────────────────────
    print("\nComputing metrics...")

    # Metric 1: Vehicle count
    vehicle_count = compute_vehicle_count(detections)
    with open(output_dir / "vehicle_count.json", "w") as f:
        json.dump(vehicle_count, f, indent=2)
    print(f"  Vehicle count: {vehicle_count['total_unique']} unique vehicles")

    if H is not None:
        # Load parking spaces
        print("  Loading parking spaces...")
        parking_spaces = get_parking_spaces()
        print(f"  {len(parking_spaces)} parking spaces loaded")

        # Metric 2: Occupancy timeline
        occupancy = compute_occupancy_timeline(
            detections, H, parking_spaces, sample_interval=args.sample_interval
        )
        with open(output_dir / "occupancy_timeline.json", "w") as f:
            json.dump(occupancy, f, indent=2)
        if occupancy["occupied"]:
            avg_occ = round(np.mean(occupancy["occupied"]), 1)
            print(f"  Occupancy: avg {avg_occ}/{occupancy['total_spaces']} occupied")

        # Metric 3: Dwell times
        dwell = compute_dwell_times(detections, H, parking_spaces)
        with open(output_dir / "dwell_times.json", "w") as f:
            json.dump(dwell, f, indent=2)
        if dwell["stats"]:
            print(
                f"  Dwell times: mean {dwell['stats']['mean_sec']}s, "
                f"median {dwell['stats']['median_sec']}s ({dwell['stats']['count']} events)"
            )

        # Metric 4: Entry/exit
        entry_exit = compute_entry_exit(detections, H)
        with open(output_dir / "entry_exit.json", "w") as f:
            json.dump(entry_exit, f, indent=2)
        print(f"  Entries: {entry_exit['entry_count']}, Exits: {entry_exit['exit_count']}")
    else:
        print("  Skipping spatial metrics (no homography available)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Congestion Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\nRunning congestion zone analysis...")
    try:
        analyzer = CongestionAnalyzer(
            base_dir=str(PROJECT_ROOT),
            scene_name=scene_name,
            frame_w=3840,
            frame_h=2160,
            grid_rows=4,
            grid_cols=4,
        )
        congestion_results = analyzer.analyze()
        print(f"  Congestion data saved to {output_dir / 'dashboard_data'}/")
    except FileNotFoundError as exc:
        print(f"  WARNING: Congestion analysis skipped — {exc}")
    except Exception as exc:
        print(f"  WARNING: Congestion analysis failed — {exc}")
        traceback.print_exc()

    print(f"\n All results saved to {output_dir}/")


if __name__ == "__main__":
    main() 