"""
Prepare DLP video + XML annotations as a YOLO test dataset.

Extracts frames from the video at a configurable stride, converts
bounding boxes from the XML to YOLO format using VisDrone-compatible class IDs.
This test set is used to evaluate a VisDrone-trained model on DLP parking footage.

Usage:
    python -m src.detection.prepare_dlp_dataset
    python -m src.detection.prepare_dlp_dataset --stride 50
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Map DLP vehicle types to VisDrone class IDs so model.val() works directly
# VisDrone classes: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van, 5=truck,
#                   6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
DLP_TO_VISDRONE = {"Car": 3, "Medium Vehicle": 4, "Bus": 8}

# Full VisDrone class names (needed for dataset.yaml)
VISDRONE_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}


def extract_test_set(
    video_path: str,
    xml_path: str,
    output_dir: str,
    stride: int = 25,
):
    """
    Extract frames and YOLO labels from DLP video + XML as a test-only dataset.

    Args:
        video_path: Path to the .MOV video file.
        xml_path: Path to the _data.xml annotation file.
        output_dir: Output directory for the YOLO dataset.
        stride: Extract every N-th frame (default 25 → ~452 images at 1/sec from 11309 frames).
    """
    output = Path(output_dir)
    (output / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "test").mkdir(parents=True, exist_ok=True)

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
    print(f"Extracting {len(frame_ids)} test frames (stride={stride})")

    extracted = 0

    for frame_idx in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx not in frame_elements or frame_idx % stride != 0:
            continue

        img_name = f"frame_{frame_idx:06d}.jpg"

        # Save image
        cv2.imwrite(str(output / "images" / "test" / img_name), frame)

        # Convert annotations to YOLO format with VisDrone class IDs
        label_lines = []
        frame_el = frame_elements[frame_idx]

        for traj in frame_el.findall("trajectory"):
            vtype = traj.get("type")
            if vtype not in DLP_TO_VISDRONE:
                continue

            cls_id = DLP_TO_VISDRONE[vtype]

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
        with open(output / "labels" / "test" / label_name, "w") as f:
            f.write("\n".join(label_lines) + "\n" if label_lines else "")

        extracted += 1

    cap.release()
    print(f"Extracted {extracted} test frames")

    # Write dataset YAML with VisDrone class names
    dataset_yaml = {
        "path": str(output.resolve()),
        "test": "images/test",
        "names": VISDRONE_NAMES,
    }
    yaml_path = output / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    print(f"Dataset config written to {yaml_path}")

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare DLP test set for YOLO evaluation")
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data/raw/DLP/raw/DJI_0012.MOV"))
    parser.add_argument("--xml", default=str(PROJECT_ROOT / "data/raw/DLP/raw/DJI_0012_data.xml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/processed/dlp_yolo_dataset"))
    parser.add_argument("--stride", type=int, default=25,
                        help="Extract every N-th frame (default: 25, ~1 per second)")
    args = parser.parse_args()

    extract_test_set(
        video_path=args.video,
        xml_path=args.xml,
        output_dir=args.output,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
