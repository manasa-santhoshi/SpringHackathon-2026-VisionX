"""
CLI entry point for the parking analytics pipeline.

Usage:
    python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV
    python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --model models/yolo11-visdrone/weights/bestVisDrone.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path for DLP submodule
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))

from dlp.visualizer import Visualizer
from dlp.dataset import Dataset

from src.detection.yolo_detector import YOLODetector
from src.pipeline.homography import compute_homography, load_homography, save_homography
from src.pipeline.metrics import (
    compute_dwell_times,
    compute_entry_exit,
    compute_occupancy_timeline,
    compute_person_count,
    compute_psi,
    compute_vehicle_count,
)


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
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models/yolo11n-visdrone/weights/bestVisDrone.pt"),
                        help="YOLO model path (default: VisDrone-finetuned model)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1920, help="Inference image size")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="Occupancy sampling interval in seconds")
    parser.add_argument("--grid", nargs=2, type=int, default=[4, 4],
                        metavar=("COLS", "ROWS"),
                        help="Grid divisions for PSI metric (default: 4 4)")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip detection, reuse existing detections.json")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    scene_name = get_scene_name(args.video)

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "processed" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Homography ---
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

    # --- Step 2: Detection + Tracking ---
    detections_path = output_dir / "detections.json"

    if args.skip_detection:
        if not detections_path.exists():
            print(f"ERROR: --skip-detection but {detections_path} not found.")
            print("Run without --skip-detection first.")
            sys.exit(1)
        print(f"Loading cached detections from {detections_path}")
        with open(detections_path) as f:
            detections_data = json.load(f)
        # Rebuild FrameDetections from serialized data
        from src.detection.base import DetectedVehicle, FrameDetections
        detections = []
        for fd in detections_data:
            vehicles = [
                DetectedVehicle(
                    track_id=v["track_id"], bbox=tuple(v["bbox"]),
                    confidence=v["confidence"], class_name=v["class_name"],
                    center_px=tuple(v["center_px"]),
                )
                for v in fd.get("vehicles", [])
            ]
            persons = [
                DetectedVehicle(
                    track_id=p["track_id"], bbox=tuple(p["bbox"]),
                    confidence=p["confidence"], class_name=p["class_name"],
                    center_px=tuple(p["center_px"]),
                )
                for p in fd.get("persons", [])
            ]
            detections.append(FrameDetections(
                frame_idx=fd["frame_idx"], timestamp=fd["timestamp"],
                vehicles=vehicles, persons=persons,
            ))
        print(f"Loaded {len(detections)} frames from cache")
    else:
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
                "persons": [
                    {
                        "track_id": p.track_id,
                        "bbox": list(p.bbox),
                        "confidence": p.confidence,
                        "class_name": p.class_name,
                        "center_px": list(p.center_px),
                    }
                    for p in frame.persons
                ],
            })
        with open(detections_path, "w") as f:
            json.dump(detections_data, f)
        print(f"Saved detections to {detections_path}")

    # --- Step 3: Compute metrics ---
    print("\nComputing metrics...")

    # Metric 1: Vehicle count (vehicles only — persons excluded)
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
            print(f"  Dwell times: mean {dwell['stats']['mean_sec']}s, "
                  f"median {dwell['stats']['median_sec']}s ({dwell['stats']['count']} events)")

        # Metric 4: Entry/exit
        entry_exit = compute_entry_exit(detections, H)
        with open(output_dir / "entry_exit.json", "w") as f:
            json.dump(entry_exit, f, indent=2)
        print(f"  Entries: {entry_exit['entry_count']}, Exits: {entry_exit['exit_count']}")
    else:
        print("  Skipping spatial metrics (no homography available)")

    # --- Step 4: Pedestrian metrics (no homography needed) ---
    print("\nComputing pedestrian metrics...")

    person_count = compute_person_count(detections)
    with open(output_dir / "person_count.json", "w") as f:
        json.dump(person_count, f, indent=2)
    print(f"  Person count: {person_count['total_unique']} unique persons detected")

    psi = compute_psi(detections_data, grid_cols=args.grid[0], grid_rows=args.grid[1])
    with open(output_dir / "psi.json", "w") as f:
        json.dump(psi, f, indent=2)
    print(f"  PSI computed for {len(psi['zones'])} zones "
          f"({args.grid[0]}x{args.grid[1]} grid)")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
