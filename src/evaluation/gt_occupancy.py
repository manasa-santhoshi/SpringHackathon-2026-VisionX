"""
Compute ground-truth occupancy timeline from DLP annotations.

Uses DLP instance positions (ground coords) to check which parking spaces
are occupied at each annotated frame.

Usage:
    python -m src.evaluation.gt_occupancy --scene DJI_0012
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

from src.pipeline.metrics import _point_in_rect
from src.pipeline.run import get_parking_spaces


def compute_gt_occupancy(
    ds: Dataset,
    scene_token: str,
    parking_spaces,
    sample_interval: float = 1.0,
) -> dict:
    """Compute ground-truth occupancy timeline from DLP annotations."""
    scene = ds.get("scene", scene_token)
    total_spaces = len(parking_spaces)
    area_totals = parking_spaces.groupby("area").size().to_dict()
    by_area = {area: {"occupied": [], "total": int(total)} for area, total in area_totals.items()}

    # Precompute parking space data for faster lookup
    spaces = []
    for _, row in parking_spaces.iterrows():
        spaces.append({
            "id": row["id"],
            "area": row["area"],
            "corners": row.iloc[2:10].to_numpy(),
        })

    # Also count static obstacles that are parked
    obstacle_positions = []
    for obs_token in scene.get("obstacles", []):
        obs = ds.get("obstacle", obs_token)
        obstacle_positions.append(obs["coords"])

    timestamps = []
    occupied_counts = []
    free_counts = []
    last_sampled_time = -sample_interval

    frame_token = scene["first_frame"]
    frame_idx = 0

    while frame_token:
        frame = ds.get("frame", frame_token)
        timestamp = frame["timestamp"]

        if timestamp - last_sampled_time >= sample_interval:
            last_sampled_time = timestamp

            # Collect all vehicle positions in this frame
            positions = []
            for inst_token in frame["instances"]:
                inst = ds.get("instance", inst_token)
                agent = ds.get("agent", inst["agent_token"])
                if agent["type"] not in {"Pedestrian", "Undefined", "Bicycle"}:
                    positions.append(inst["coords"])

            # Add static obstacles
            positions.extend(obstacle_positions)

            # Check which spaces are occupied
            occupied_spaces = set()
            area_occupied = defaultdict(int)

            for gx, gy in positions:
                for space in spaces:
                    if _point_in_rect(gx, gy, space["corners"]):
                        if space["id"] not in occupied_spaces:
                            occupied_spaces.add(space["id"])
                            area_occupied[space["area"]] += 1
                        break

            n_occupied = len(occupied_spaces)
            timestamps.append(round(timestamp, 2))
            occupied_counts.append(n_occupied)
            free_counts.append(total_spaces - n_occupied)
            for area in by_area:
                by_area[area]["occupied"].append(area_occupied.get(area, 0))

        frame_token = frame["next"]
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  Frame {frame_idx}")

    return {
        "timestamps": timestamps,
        "occupied": occupied_counts,
        "free": free_counts,
        "total_spaces": total_spaces,
        "by_area": by_area,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute GT occupancy from DLP annotations")
    parser.add_argument("--scene", default="DJI_0012", help="Scene name")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="Seconds between occupancy samples")
    args = parser.parse_args()

    # Load DLP dataset
    print(f"Loading DLP ground truth for {args.scene}...")
    ds = Dataset()
    data_dir = PROJECT_ROOT / "data" / "raw" / "DLP" / "json"
    ds.load(str(data_dir / args.scene))
    scene_token = list(ds.scenes.keys())[0]

    # Load parking spaces
    print("Loading parking spaces...")
    parking_spaces = get_parking_spaces()
    print(f"  {len(parking_spaces)} spaces loaded")

    # Compute GT occupancy
    print("Computing ground-truth occupancy...")
    result = compute_gt_occupancy(
        ds, scene_token, parking_spaces, sample_interval=args.sample_interval
    )

    # Save
    output_dir = PROJECT_ROOT / "data" / "processed" / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gt_occupancy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    avg_occ = round(np.mean(result["occupied"]), 1) if result["occupied"] else 0
    print(f"\nDone! {len(result['timestamps'])} samples")
    print(f"  Avg occupancy: {avg_occ}/{result['total_spaces']}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
