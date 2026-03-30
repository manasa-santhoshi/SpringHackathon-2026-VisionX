"""
Frame-difference occupancy counter baseline.

A lightweight, GPU-free method that detects parking space occupancy by
comparing each frame against a reference frame using pixel subtraction.

Usage:
    python -m src.baseline.frame_diff --video data/raw/DLP/raw/DJI_0012.MOV
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))

from src.pipeline.homography import compute_homography, load_homography, save_homography
from src.pipeline.run import get_parking_spaces


def ground_to_pixel(H_inv: np.ndarray, corners_ground: np.ndarray) -> np.ndarray:
    """Convert ground-coordinate corners (4, 2) to pixel coordinates via inverse homography."""
    ones = np.ones((len(corners_ground), 1))
    pts_h = np.hstack([corners_ground, ones])  # (4, 3)
    pts_px = (H_inv @ pts_h.T).T  # (4, 3)
    pts_px = pts_px[:, :2] / pts_px[:, 2:3]
    return pts_px.astype(np.int32)


def build_space_masks(
    parking_spaces, H_inv: np.ndarray, frame_shape: tuple[int, int]
) -> list[dict]:
    """Pre-compute a binary pixel mask for each parking space."""
    masks = []
    for _, row in parking_spaces.iterrows():
        coords = row.iloc[2:10].to_numpy().reshape(4, 2)
        pixel_corners = ground_to_pixel(H_inv, coords)
        mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_corners], 1)
        pixel_count = int(mask.sum())
        if pixel_count > 0:
            masks.append({
                "id": row["id"],
                "area": row["area"],
                "mask": mask,
                "pixel_count": pixel_count,
            })
    return masks


def compute_baseline_occupancy(
    video_path: str,
    H: np.ndarray,
    parking_spaces,
    ref_frame_idx: int = 0,
    threshold: int = 30,
    occupancy_ratio: float = 0.25,
    sample_interval: float = 1.0,
) -> dict:
    """Run frame-difference occupancy detection on a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read reference frame {ref_frame_idx}")
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    # Build per-space masks
    H_inv = np.linalg.inv(H)
    space_masks = build_space_masks(parking_spaces, H_inv, ref_gray.shape)
    total_spaces = len(space_masks)

    area_totals = parking_spaces.groupby("area").size().to_dict()
    by_area = {area: {"occupied": [], "total": int(total)} for area, total in area_totals.items()}

    timestamps = []
    occupied_counts = []
    free_counts = []
    last_sampled_time = -sample_interval

    # Process video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        if timestamp - last_sampled_time >= sample_interval:
            last_sampled_time = timestamp
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            diff = cv2.absdiff(frame_gray, ref_gray)
            binary = (diff > threshold).astype(np.uint8)

            area_occupied = defaultdict(int)
            n_occupied = 0

            for space in space_masks:
                changed = int(np.sum(binary * space["mask"]))
                ratio = changed / space["pixel_count"]
                if ratio > occupancy_ratio:
                    n_occupied += 1
                    area_occupied[space["area"]] += 1

            timestamps.append(round(timestamp, 2))
            occupied_counts.append(n_occupied)
            free_counts.append(total_spaces - n_occupied)
            for area in by_area:
                by_area[area]["occupied"].append(area_occupied.get(area, 0))

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()

    return {
        "timestamps": timestamps,
        "occupied": occupied_counts,
        "free": free_counts,
        "total_spaces": total_spaces,
        "by_area": by_area,
    }


def main():
    parser = argparse.ArgumentParser(description="Frame-difference occupancy baseline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--ref-frame", type=int, default=0, help="Reference frame index")
    parser.add_argument("--threshold", type=int, default=30, help="Pixel diff threshold")
    parser.add_argument("--occupancy-ratio", type=float, default=0.25,
                        help="Changed-pixel ratio to declare occupied")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="Seconds between occupancy samples")
    parser.add_argument("--output", help="Output JSON path (auto-detected if omitted)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    scene_name = video_path.stem

    # Load or compute homography
    output_dir = PROJECT_ROOT / "data" / "processed" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    homography_path = output_dir / "homography.npy"
    xml_path = video_path.parent / f"{scene_name}_data.xml"

    if homography_path.exists():
        print(f"Loading homography from {homography_path}")
        H = load_homography(str(homography_path))
    elif xml_path.exists():
        print(f"Computing homography from {xml_path}")
        H = compute_homography(str(xml_path))
        save_homography(H, str(homography_path))
    else:
        raise FileNotFoundError(
            f"No homography or XML annotation found for {scene_name}. "
            "Run the main pipeline first to generate homography.npy."
        )

    # Load parking spaces
    print("Loading parking spaces...")
    parking_spaces = get_parking_spaces()
    print(f"  {len(parking_spaces)} spaces loaded")

    # Run baseline
    print(f"Running frame-difference baseline on {video_path}")
    result = compute_baseline_occupancy(
        str(video_path),
        H,
        parking_spaces,
        ref_frame_idx=args.ref_frame,
        threshold=args.threshold,
        occupancy_ratio=args.occupancy_ratio,
        sample_interval=args.sample_interval,
    )

    # Save output
    out_path = Path(args.output) if args.output else output_dir / "baseline_occupancy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    avg_occ = round(np.mean(result["occupied"]), 1) if result["occupied"] else 0
    print(f"\nDone! {len(result['timestamps'])} samples")
    print(f"  Avg occupancy: {avg_occ}/{result['total_spaces']}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
