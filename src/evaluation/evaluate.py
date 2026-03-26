"""
Evaluation module: compare YOLO detections against DLP ground truth.

Loads the DLP JSON ground truth for a scene and compares it with model detections
to compute detection accuracy metrics.

Usage:
    python -m src.evaluation.evaluate --scene DJI_0012
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))

from dlp.dataset import Dataset
from dlp.visualizer import Visualizer

from src.pipeline.homography import load_homography, pixel_to_ground


def load_detections(processed_dir: Path) -> list[dict]:
    """Load saved detections from JSON."""
    with open(processed_dir / "detections.json") as f:
        return json.load(f)


def load_ground_truth(scene_name: str) -> tuple[Dataset, str]:
    """Load DLP dataset for a scene."""
    ds = Dataset()
    data_dir = PROJECT_ROOT / "data" / "raw" / "DLP" / "json"
    ds.load(str(data_dir / scene_name))
    scene_token = list(ds.scene.keys())[0]
    return ds, scene_token


def count_gt_vehicles_per_frame(
    ds: Dataset, scene_token: str
) -> dict[int, int]:
    """
    Count ground-truth vehicles (non-pedestrian agents + obstacles) visible per frame.

    Returns dict mapping frame_index → vehicle count.
    """
    scene = ds.get("scene", scene_token)
    n_obstacles = len(scene["obstacles"])

    frame_counts = {}
    frame_token = scene["first_frame"]
    frame_idx = 0

    while frame_token:
        frame = ds.get("frame", frame_token)
        # Count non-pedestrian instances in this frame
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
    """
    Compare detected vehicle count vs ground truth per frame (sampled).

    Returns per-frame comparison and aggregate error metrics.
    """
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

    # GT: count non-pedestrian agents + obstacles
    gt_agents = 0
    for agent_token in scene["agents"]:
        agent = ds.get("agent", agent_token)
        if agent["type"] not in {"Pedestrian", "Undefined", "Bicycle"}:
            gt_agents += 1
    gt_obstacles = len(scene["obstacles"])
    gt_total = gt_agents + gt_obstacles

    # Detected: unique track IDs
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
    parser = argparse.ArgumentParser(description="Evaluate detections against ground truth")
    parser.add_argument("--scene", default="DJI_0012", help="Scene name (default: DJI_0012)")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.scene

    if not (processed_dir / "detections.json").exists():
        print(f"No detections found at {processed_dir}/detections.json")
        print("Run the pipeline first: python -m src.pipeline.run --video ...")
        sys.exit(1)

    print(f"Evaluating {args.scene}...")

    # Load data
    detections = load_detections(processed_dir)
    ds, scene_token = load_ground_truth(args.scene)

    # Evaluation 1: Unique vehicle count
    print("\n--- Unique Vehicle Count ---")
    unique_eval = evaluate_unique_count(detections, ds, scene_token)
    print(f"  Detected unique tracks: {unique_eval['detected_unique']}")
    print(f"  GT moving vehicles:     {unique_eval['gt_moving_vehicles']}")
    print(f"  GT static obstacles:    {unique_eval['gt_static_obstacles']}")
    print(f"  GT total:               {unique_eval['gt_total']}")

    # Evaluation 2: Per-frame vehicle count
    print("\n--- Per-Frame Vehicle Count (sampled) ---")
    gt_frame_counts = count_gt_vehicles_per_frame(ds, scene_token)
    frame_eval = evaluate_vehicle_count(detections, gt_frame_counts)
    if frame_eval["stats"]:
        s = frame_eval["stats"]
        print(f"  MAE:       {s['mae']}")
        print(f"  Median AE: {s['median_ae']}")
        print(f"  Max AE:    {s['max_ae']}")
        print(f"  Samples:   {s['n_samples']}")

    # Save evaluation results
    results = {
        "scene": args.scene,
        "unique_count": unique_eval,
        "frame_count": frame_eval,
    }
    output_path = processed_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
