"""
Prepare DLP video + XML annotations as a YOLO training dataset.

Extracts frames from the video at a configurable stride, converts
bounding boxes from the XML to YOLO format, and splits into train/val.

Usage:
    python -m src.detection.prepare_dlp_dataset
    python -m src.detection.prepare_dlp_dataset --stride 10 --val-ratio 0.2
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Map DLP types to class IDs (exclude Pedestrian and Undefined)
DLP_CLASSES = {"Car": 0, "Medium Vehicle": 1, "Bus": 2}
CLASS_NAMES = {v: k for k, v in DLP_CLASSES.items()}


def extract_dataset(
    video_path: str,
    xml_path: str,
    output_dir: str,
    stride: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Extract frames and YOLO labels from DLP video + XML.

    Args:
        video_path: Path to the .MOV video file.
        xml_path: Path to the _data.xml annotation file.
        output_dir: Output directory for the YOLO dataset.
        stride: Extract every N-th frame (default 5 → ~2260 images from 11309 frames).
        val_ratio: Fraction of frames for validation.
        test_ratio: Fraction of frames for testing.
        seed: Random seed for train/val/test split.
    """
    output = Path(output_dir)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Parse XML
    print(f"Parsing {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {total_frames} frames")

    # Collect frame IDs to extract
    frame_elements = {int(f.get("id")): f for f in root.findall("frame")}
    frame_ids = sorted(fid for fid in frame_elements if fid % stride == 0)
    print(f"Extracting {len(frame_ids)} frames (stride={stride})")

    # Train/val/test split
    random.seed(seed)
    random.shuffle(frame_ids)
    n_test = max(1, int(len(frame_ids) * test_ratio))
    n_val = max(1, int(len(frame_ids) * val_ratio))
    test_ids = set(frame_ids[:n_test])
    val_ids = set(frame_ids[n_test:n_test + n_val])
    train_ids = set(frame_ids[n_test + n_val:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Extract frames and labels
    frame_idx = 0
    extracted = 0

    for frame_idx in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx not in frame_elements or frame_idx % stride != 0:
            continue

        if frame_idx in test_ids:
            split = "test"
        elif frame_idx in val_ids:
            split = "val"
        else:
            split = "train"
        img_name = f"frame_{frame_idx:06d}.jpg"

        # Save image
        cv2.imwrite(str(output / "images" / split / img_name), frame)

        # Convert annotations to YOLO format
        label_lines = []
        frame_el = frame_elements[frame_idx]

        for traj in frame_el.findall("trajectory"):
            vtype = traj.get("type")
            if vtype not in DLP_CLASSES:
                continue

            cls_id = DLP_CLASSES[vtype]

            # Get bounding box from corners
            fl_x = float(traj.get("front_left_x"))
            fl_y = float(traj.get("front_left_y"))
            fr_x = float(traj.get("front_right_x"))
            fr_y = float(traj.get("front_right_y"))
            rl_x = float(traj.get("rear_left_x"))
            rl_y = float(traj.get("rear_left_y"))
            rr_x = float(traj.get("rear_right_x"))
            rr_y = float(traj.get("rear_right_y"))

            # Axis-aligned bounding box from the 4 corners
            all_x = [fl_x, fr_x, rl_x, rr_x]
            all_y = [fl_y, fr_y, rl_y, rr_y]
            x_min = max(0, min(all_x))
            x_max = min(width, max(all_x))
            y_min = max(0, min(all_y))
            y_max = min(height, max(all_y))

            # Skip degenerate boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            # YOLO format: class cx cy w h (normalized)
            cx = ((x_min + x_max) / 2) / width
            cy = ((y_min + y_max) / 2) / height
            bw = (x_max - x_min) / width
            bh = (y_max - y_min) / height

            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Save label
        label_name = f"frame_{frame_idx:06d}.txt"
        with open(output / "labels" / split / label_name, "w") as f:
            f.write("\n".join(label_lines) + "\n" if label_lines else "")

        extracted += 1

    cap.release()
    print(f"Extracted {extracted} frames")

    # Write dataset YAML
    dataset_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: k for k, v in DLP_CLASSES.items()},
    }
    yaml_path = output / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"Dataset config written to {yaml_path}")

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare DLP data for YOLO training")
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data/raw/DLP/raw/DJI_0012.MOV"))
    parser.add_argument("--xml", default=str(PROJECT_ROOT / "data/raw/DLP/raw/DJI_0012_data.xml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/dlp_yolo_dataset"))
    parser.add_argument("--stride", type=int, default=5,
                        help="Extract every N-th frame (default: 5)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    extract_dataset(
        video_path=args.video,
        xml_path=args.xml,
        output_dir=args.output,
        stride=args.stride,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
