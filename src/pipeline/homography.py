"""
Homography computation for mapping video pixel coordinates to local parking lot ground coordinates.

Uses the DLP XML annotation file which provides both pixel bounding box corners and UTM coordinates
for each vehicle at each frame. The UTM→local conversion uses ORIGIN from the DLP dataset.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# UTM origin used by DLP dataset to convert UTM → local ground coords
# From dlp-dataset/raw-data-processing/generate_tokens.py
UTM_ORIGIN = {"x": 747064, "y": 3856846}


def parse_xml_correspondences(
    xml_path: str, max_frames: int = 50, stride: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse the DLP XML annotation file and extract pixel↔ground coordinate pairs.

    Samples vehicles from frames at the given stride to get spatially diverse points.
    Uses bounding box centers in pixel space and UTM→local conversion for ground coords.

    Returns:
        pixel_pts: (N, 2) array of pixel coordinates
        ground_pts: (N, 2) array of local ground coordinates
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pixel_pts = []
    ground_pts = []

    frames_sampled = 0
    for frame_el in root.findall("frame"):
        frame_id = int(frame_el.get("id"))
        if frame_id % stride != 0:
            continue

        for traj in frame_el.findall("trajectory"):
            # Pixel bounding box corners → center
            fl_x = float(traj.get("front_left_x"))
            fl_y = float(traj.get("front_left_y"))
            fr_x = float(traj.get("front_right_x"))
            fr_y = float(traj.get("front_right_y"))
            rl_x = float(traj.get("rear_left_x"))
            rl_y = float(traj.get("rear_left_y"))
            rr_x = float(traj.get("rear_right_x"))
            rr_y = float(traj.get("rear_right_y"))

            center_px = ((fl_x + fr_x + rl_x + rr_x) / 4,
                         (fl_y + fr_y + rl_y + rr_y) / 4)

            # UTM → local ground coords
            utm_x = float(traj.get("utm_x"))
            utm_y = float(traj.get("utm_y"))
            local_x = UTM_ORIGIN["x"] - utm_x
            local_y = UTM_ORIGIN["y"] - utm_y

            pixel_pts.append(center_px)
            ground_pts.append((local_x, local_y))

        frames_sampled += 1
        if frames_sampled >= max_frames:
            break

    return np.array(pixel_pts, dtype=np.float64), np.array(ground_pts, dtype=np.float64)


def compute_homography(
    xml_path: str, max_frames: int = 50, stride: int = 10
) -> np.ndarray:
    """
    Compute the homography matrix mapping video pixel coords → local ground coords.

    Uses RANSAC for robustness against annotation noise.

    Returns:
        H: (3, 3) homography matrix
    """
    pixel_pts, ground_pts = parse_xml_correspondences(xml_path, max_frames, stride)

    H, mask = cv2.findHomography(pixel_pts, ground_pts, cv2.RANSAC, 5.0)

    inliers = mask.ravel().sum()
    total = len(mask)
    print(f"Homography computed: {inliers}/{total} inliers")

    return H


def pixel_to_ground(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Transform a single pixel point to ground coordinates using homography H."""
    pt = np.array([px, py, 1.0])
    result = H @ pt
    result /= result[2]
    return float(result[0]), float(result[1])


def pixels_to_ground(H: np.ndarray, pixel_pts: np.ndarray) -> np.ndarray:
    """
    Transform multiple pixel points to ground coordinates.

    Args:
        H: (3, 3) homography matrix
        pixel_pts: (N, 2) array of pixel coordinates

    Returns:
        (N, 2) array of ground coordinates
    """
    n = len(pixel_pts)
    ones = np.ones((n, 1))
    pts_h = np.hstack([pixel_pts, ones])  # (N, 3)
    result = (H @ pts_h.T).T  # (N, 3)
    result /= result[:, 2:3]
    return result[:, :2]


def save_homography(H: np.ndarray, path: str) -> None:
    """Save homography matrix to a .npy file."""
    np.save(path, H)


def load_homography(path: str) -> np.ndarray:
    """Load homography matrix from a .npy file."""
    return np.load(path)
