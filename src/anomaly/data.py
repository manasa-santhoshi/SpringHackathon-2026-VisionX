"""
CHAD dataset loader for skeleton-based anomaly detection.

Loads pre-extracted skeleton annotations (.pkl) and frame-level anomaly labels (.npy)
from the CHAD dataset and creates sliding-window sequences for training/testing.

CHAD annotation format (per .pkl file):
    {frame_number: {person_id: [np.array(bbox_xywh), np.array(keypoints_xyc)]}}

Keypoints: 17 COCO joints, each with (x, y, confidence) = 51 values total.
Labels (.npy): binary array, 0=normal, 1=anomalous, one per frame.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 17 COCO keypoints × 2 (x, y only, drop confidence)
NUM_KEYPOINTS = 17
FEATURES_PER_KEYPOINT = 2
POSE_DIM = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT  # 34
# With velocity: position (34) + velocity (34) = 68
INPUT_DIM = POSE_DIM * 2  # 68

# Max allowed gap between consecutive frames in a track segment
MAX_FRAME_GAP = 5


def load_chad_annotation(pkl_path: Path) -> dict:
    """Load a single CHAD .pkl annotation file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_chad_labels(npy_path: Path) -> np.ndarray:
    """Load frame-level anomaly labels."""
    return np.load(npy_path)


def extract_person_tracks(
    annotation: dict,
    normalize: bool = True,
) -> dict[int, tuple[list[np.ndarray], list[int]]]:
    """
    Extract per-person skeleton tracks from a CHAD annotation dict.

    Returns:
        {person_id: (skeletons, frame_indices)} where each skeleton is (34,)
        array of x,y coords and frame_indices are the original frame numbers.
    """
    person_tracks: dict[int, dict[int, np.ndarray]] = {}

    for frame_num, persons in annotation.items():
        frame_idx = int(frame_num)
        for person_id, (bbox, keypoints) in persons.items():
            pid = int(person_id)
            if pid not in person_tracks:
                person_tracks[pid] = {}

            kp = np.array(keypoints, dtype=np.float32)
            # keypoints are flat: [x1,y1,c1, x2,y2,c2, ...] = 51 values
            if kp.size == 51:
                kp = kp.reshape(17, 3)
            elif kp.size == 34:
                kp = kp.reshape(17, 2)
            else:
                continue

            # Extract only x, y (drop confidence if present)
            xy = kp[:, :2].copy()

            if normalize:
                # Normalize relative to bounding box center and size
                bbox = np.array(bbox, dtype=np.float32)
                bx, by, bw, bh = bbox[:4]
                if bw > 0 and bh > 0:
                    cx, cy = bx + bw / 2, by + bh / 2
                    xy[:, 0] = (xy[:, 0] - cx) / bw
                    xy[:, 1] = (xy[:, 1] - cy) / bh

            person_tracks[pid][frame_idx] = xy.flatten()

    # Convert to sorted frame order
    result = {}
    for pid, frames in person_tracks.items():
        sorted_frames = sorted(frames.keys())
        result[pid] = (
            [frames[f] for f in sorted_frames],
            sorted_frames,
        )

    return result


def split_track_on_gaps(
    skeletons: list[np.ndarray],
    frame_indices: list[int],
    max_gap: int = MAX_FRAME_GAP,
) -> list[tuple[list[np.ndarray], list[int]]]:
    """
    Split a person track into continuous segments where consecutive
    frames are at most max_gap apart. This avoids creating sequences
    that span large temporal discontinuities.
    """
    if len(skeletons) <= 1:
        return [(skeletons, frame_indices)]

    segments = []
    seg_skels = [skeletons[0]]
    seg_frames = [frame_indices[0]]

    for i in range(1, len(skeletons)):
        gap = frame_indices[i] - frame_indices[i - 1]
        if gap > max_gap:
            # Start new segment
            segments.append((seg_skels, seg_frames))
            seg_skels = [skeletons[i]]
            seg_frames = [frame_indices[i]]
        else:
            seg_skels.append(skeletons[i])
            seg_frames.append(frame_indices[i])

    segments.append((seg_skels, seg_frames))
    return segments


def add_velocity_features(poses: np.ndarray) -> np.ndarray:
    """
    Add velocity (frame-to-frame differences) as additional features.

    Args:
        poses: (seq_len, 34) array of keypoint positions.

    Returns:
        (seq_len, 68) array of [position, velocity] per frame.
    """
    velocity = np.zeros_like(poses)
    velocity[1:] = poses[1:] - poses[:-1]
    # First frame velocity = 0 (no previous frame)
    return np.concatenate([poses, velocity], axis=-1)


def load_split_file(split_path: Path) -> list[str]:
    """Load a CHAD split file (list of video filenames)."""
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


class CHADSkeletonDataset(Dataset):
    """
    PyTorch dataset for CHAD skeleton sequences.

    Creates sliding-window sequences from person skeleton tracks.
    For training (unsupervised): only normal sequences.
    For testing: all sequences with corresponding labels.
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        split_num: int = 1,
        cameras: list[int] | None = None,
        seq_len: int = 12,
        stride: int = 1,
        normalize: bool = True,
        pred_len: int = 6,
        use_velocity: bool = True,
    ):
        """
        Args:
            data_root: Path to CHAD_Meta directory.
            split: "train" or "test".
            split_num: 1 (unsupervised) or 2 (supervised).
            cameras: List of camera IDs to use (default: all 4).
            seq_len: Number of frames per input sequence.
            stride: Stride between consecutive sequences.
            normalize: Whether to normalize keypoints to bbox-relative coords.
            pred_len: Number of future frames to predict.
            use_velocity: Whether to add velocity features.
        """
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.pred_len = pred_len
        self.cameras = cameras or [1, 2, 3, 4]
        self.use_velocity = use_velocity

        # Load split file
        split_file = self.data_root / "splits" / f"{split}_split_{split_num}.txt"
        video_names = load_split_file(split_file)

        # Filter by camera
        video_names = [
            v for v in video_names
            if int(v.split("_")[0]) in self.cameras
        ]

        self.sequences: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.video_ids: list[str] = []
        self.camera_ids: list[int] = []

        annotations_dir = self.data_root / "annotations"
        labels_dir = self.data_root / "anomaly_labels"

        for video_name in video_names:
            stem = video_name.replace(".mp4", "")
            pkl_path = annotations_dir / f"{stem}.pkl"
            npy_path = labels_dir / f"{stem}.npy"

            if not pkl_path.exists() or not npy_path.exists():
                continue

            annotation = load_chad_annotation(pkl_path)
            frame_labels = load_chad_labels(npy_path)
            camera_id = int(stem.split("_")[0])

            # Extract per-person tracks (with frame indices for label mapping)
            tracks = extract_person_tracks(annotation, normalize=normalize)

            for pid, (skeletons, frame_indices) in tracks.items():
                # Split track on temporal gaps to avoid discontinuities
                segments = split_track_on_gaps(skeletons, frame_indices)

                for seg_skels, seg_frames in segments:
                    total_len = seq_len + pred_len
                    if len(seg_skels) < total_len:
                        continue

                    for i in range(0, len(seg_skels) - total_len + 1, stride):
                        poses = np.stack(seg_skels[i : i + total_len], axis=0)

                        if use_velocity:
                            seq = add_velocity_features(poses)
                        else:
                            seq = poses

                        self.sequences.append(
                            torch.tensor(seq, dtype=torch.float32)
                        )

                        # Label: anomalous if any frame in window is anomalous
                        window_frames = seg_frames[i : i + total_len]
                        valid_frames = [
                            f for f in window_frames if f < len(frame_labels)
                        ]
                        if valid_frames:
                            has_anomaly = int(frame_labels[valid_frames].max())
                        else:
                            has_anomaly = 0
                        self.labels.append(has_anomaly)
                        self.video_ids.append(stem)
                        self.camera_ids.append(camera_id)

        # For unsupervised training split, filter out anomalous sequences
        if split == "train" and split_num == 1:
            normal_mask = [l == 0 for l in self.labels]
            self.sequences = [s for s, m in zip(self.sequences, normal_mask) if m]
            self.labels = [l for l, m in zip(self.labels, normal_mask) if m]
            self.video_ids = [v for v, m in zip(self.video_ids, normal_mask) if m]
            self.camera_ids = [c for c, m in zip(self.camera_ids, normal_mask) if m]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        return {
            "input": seq[: self.seq_len],          # (seq_len, input_dim)
            "target": seq[self.seq_len :],          # (pred_len, input_dim)
            "label": self.labels[idx],
            "video_id": self.video_ids[idx],
            "camera_id": self.camera_ids[idx],
        }
