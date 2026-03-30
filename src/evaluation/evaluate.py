"""
Evaluation module: test VisDrone-trained YOLO model on DLP parking lot data.

Two evaluation modes:
1. YOLO val: Run model.val() on the DLP test set for standard mAP/precision/recall
2. Pipeline eval: Compare pipeline detections vs DLP JSON ground truth for vehicle counts

Usage:
    python -m src.evaluation.evaluate --scene DJI_0012
    python -m src.evaluation.evaluate --scene DJI_0012 --model models/yolo11n-visdrone/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))

from dlp.dataset import Dataset


# DLP ground truth only has these VisDrone classes annotated
DLP_EVAL_CLASSES = [3, 4, 8]  # car, van, bus


def run_yolo_val(model_path: str, dataset_yaml: str) -> dict:
    """
    Run YOLO validation on the DLP test set.

    Only evaluates on classes with ground truth labels (car, van, bus).
    Returns standard detection metrics (mAP50, mAP50-95, precision, recall).
    """
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, split="test", classes=DLP_EVAL_CLASSES, verbose=True)

    return {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50-95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "per_class": {
            name: {
                "mAP50": round(float(results.box.maps[i]), 4) if i < len(results.box.maps) else None,
            }
            for i, name in enumerate(results.names.values())
            if i < len(results.box.maps)
        },
    }


def load_detections(processed_dir: Path) -> list[dict]:
    """Load saved pipeline detections from JSON."""
    det_path = processed_dir / "detections.json"
    if not det_path.exists():
        return []
    with open(det_path) as f:
        return json.load(f)


def load_ground_truth(scene_name: str) -> tuple[Dataset, str]:
    """Load DLP dataset for a scene."""
    ds = Dataset()
    data_dir = PROJECT_ROOT / "data" / "raw" / "DLP" / "json"
    ds.load(str(data_dir / scene_name))
    scene_token = list(ds.scene.keys())[0]
    return ds, scene_token


def count_gt_vehicles_per_frame(ds: Dataset, scene_token: str) -> dict[int, int]:
    """
    Count ground-truth vehicles (non-pedestrian agents + obstacles) visible per frame.
    """
    scene = ds.get("scene", scene_token)
    n_obstacles = len(scene["obstacles"])

    frame_counts = {}
    frame_token = scene["first_frame"]
    frame_idx = 0

    while frame_token:
        frame = ds.get("frame", frame_token)
        n_agents = 0
        for inst_token in frame["instances"]:
            inst = ds.get("instance", inst_token)
            agent = ds.get("agent", inst["agent_token"])
            if agent["type"] not in {"Pedestrian", "Undefined", "Bicycle"}:
                n_agents += 1

        frame_counts[frame_idx] = n_agents + n_obstacles
        frame_token = frame["next"]
        frame_idx += 1

    return frame_counts


def evaluate_vehicle_count(
    detections: list[dict], gt_frame_counts: dict[int, int], sample_interval: int = 25
) -> dict:
    """Compare detected vehicle count vs ground truth per frame (sampled)."""
    comparisons = []
    errors = []

    for det_frame in detections:
        frame_idx = det_frame["frame_idx"]
        if frame_idx % sample_interval != 0:
            continue
        if frame_idx not in gt_frame_counts:
            continue

        detected = len(det_frame["vehicles"])
        gt = gt_frame_counts[frame_idx]
        error = detected - gt

        comparisons.append({
            "frame_idx": frame_idx,
            "timestamp": det_frame["timestamp"],
            "detected": detected,
            "ground_truth": gt,
            "error": error,
        })
        errors.append(abs(error))

    stats = {}
    if errors:
        stats = {
            "mae": round(float(np.mean(errors)), 2),
            "median_ae": round(float(np.median(errors)), 2),
            "max_ae": int(max(errors)),
            "n_samples": len(errors),
        }

    return {"comparisons": comparisons, "stats": stats}


def evaluate_unique_count(detections: list[dict], ds: Dataset, scene_token: str) -> dict:
    """Compare total unique vehicles detected vs ground truth."""
    scene = ds.get("scene", scene_token)

    gt_agents = 0
    for agent_token in scene["agents"]:
        agent = ds.get("agent", agent_token)
        if agent["type"] not in {"Pedestrian", "Undefined", "Bicycle"}:
            gt_agents += 1
    gt_obstacles = len(scene["obstacles"])
    gt_total = gt_agents + gt_obstacles

    detected_tracks = set()
    for frame in detections:
        for v in frame["vehicles"]:
            detected_tracks.add(v["track_id"])

    return {
        "detected_unique": len(detected_tracks),
        "gt_moving_vehicles": gt_agents,
        "gt_static_obstacles": gt_obstacles,
        "gt_total": gt_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on DLP data")
    parser.add_argument("--scene", default="DJI_0012", help="Scene name (default: DJI_0012)")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models/yolo11n-visdrone/weights/best.pt"),
                        help="Model path for YOLO val")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.scene
    dataset_yaml = PROJECT_ROOT / "data" / "processed" / "dlp_yolo_dataset" / "dataset.yaml"

    results = {"scene": args.scene}

    # --- YOLO val on DLP test set ---
    if dataset_yaml.exists() and Path(args.model).exists():
        print("=== YOLO Validation on DLP Test Set ===")
        yolo_metrics = run_yolo_val(args.model, str(dataset_yaml))
        results["yolo_val"] = yolo_metrics
        print(f"  mAP50:      {yolo_metrics['mAP50']}")
        print(f"  mAP50-95:   {yolo_metrics['mAP50-95']}")
        print(f"  Precision:  {yolo_metrics['precision']}")
        print(f"  Recall:     {yolo_metrics['recall']}")
    else:
        if not Path(args.model).exists():
            print(f"Model not found at {args.model}")
            print("Train first: python -m src.detection.train")
        if not dataset_yaml.exists():
            print(f"DLP test set not found at {dataset_yaml}")
            print("Prepare first: python -m src.detection.prepare_dlp_dataset")

    # --- Pipeline detection comparison ---
    detections = load_detections(processed_dir)
    if detections:
        print("\n=== Pipeline Detection vs Ground Truth ===")
        ds, scene_token = load_ground_truth(args.scene)

        # Unique vehicle count
        unique_eval = evaluate_unique_count(detections, ds, scene_token)
        results["unique_count"] = unique_eval
        print(f"  Detected unique tracks: {unique_eval['detected_unique']}")
        print(f"  GT total:               {unique_eval['gt_total']} "
              f"({unique_eval['gt_moving_vehicles']} moving + {unique_eval['gt_static_obstacles']} static)")

        # Per-frame vehicle count
        gt_frame_counts = count_gt_vehicles_per_frame(ds, scene_token)
        frame_eval = evaluate_vehicle_count(detections, gt_frame_counts)
        results["frame_count"] = frame_eval
        if frame_eval["stats"]:
            s = frame_eval["stats"]
            print(f"  MAE:       {s['mae']}")
            print(f"  Median AE: {s['median_ae']}")
            print(f"  Max AE:    {s['max_ae']}")
            print(f"  Samples:   {s['n_samples']}")
    else:
        print("\nNo pipeline detections found. Run the pipeline first:")
        print("  python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV")

    # Save results
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
