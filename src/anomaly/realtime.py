"""
Real-time anomaly detection from pre-extracted CHAD skeleton annotations.

Reads .pkl skeleton data frame-by-frame, maintains per-person sliding windows,
and runs MPED-RNN inference to produce per-frame anomaly scores.

Usage:
    from src.anomaly.realtime import RealtimeAnomalyDetector

    detector = RealtimeAnomalyDetector(model_dir="models/anomaly/cam_1_2_3_4")
    detector.load_video("1_065_0")  # loads .pkl + .npy for this video
    for frame_idx in range(detector.num_frames):
        result = detector.process_frame(frame_idx)
        # result.max_score, result.is_anomaly, result.gt_label, ...
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.anomaly.data import (
    INPUT_DIM,
    MAX_FRAME_GAP,
    POSE_DIM,
    add_velocity_features,
    extract_person_tracks,
    load_chad_annotation,
    load_chad_labels,
)
from src.anomaly.model import MPEDRNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FrameAnomalyResult:
    """Anomaly detection result for a single video frame."""

    frame_idx: int
    scores: list[float] = field(default_factory=list)  # one per person window
    max_score: float = 0.0
    mean_score: float = 0.0
    num_persons: int = 0
    is_anomaly: bool = False
    gt_label: int = 0  # 0=normal, 1=anomalous (from ground truth)


class RealtimeAnomalyDetector:
    """
    Frame-by-frame anomaly detector using pre-extracted CHAD skeletons.

    Maintains a sliding window buffer per person track. When a person's buffer
    reaches seq_len + pred_len frames, runs MPED-RNN inference and produces
    an anomaly score.
    """

    def __init__(
        self,
        model_dir: str,
        data_root: str | None = None,
        device: str | None = None,
    ):
        model_dir = Path(model_dir)

        # Load config
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)

        self.seq_len = self.config["seq_len"]
        self.pred_len = self.config["pred_len"]
        self.total_window = self.seq_len + self.pred_len

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = device

        # Load model
        self.model = MPEDRNN(
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            pred_len=self.pred_len,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=self.device)
        )
        self.model.eval()

        # Load threshold from evaluation
        eval_path = model_dir / "evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            self.threshold = eval_data.get("threshold", 0.0)
            self.threshold = 2.0
        else:
            self.threshold = 0.0

        # Data root
        if data_root is None:
            data_root = str(PROJECT_ROOT / "data" / "raw" / "CHAD" / "CHAD_Meta")
        self.data_root = Path(data_root)

        # Per-video state (set by load_video)
        self._annotation: dict | None = None
        self._labels: np.ndarray | None = None
        self._person_buffers: dict[int, list[np.ndarray]] = {}
        self._person_last_frame: dict[int, int] = {}
        self.num_frames: int = 0
        self.score_timeline: list[float] = []
        self.gt_timeline: list[int] = []

    def load_video(self, video_stem: str) -> None:
        """
        Load skeleton annotations and labels for a CHAD video.

        Args:
            video_stem: Video name without extension (e.g. "1_065_0").
        """
        pkl_path = self.data_root / "annotations" / f"{video_stem}.pkl"
        npy_path = self.data_root / "anomaly_labels" / f"{video_stem}.npy"

        self._annotation = load_chad_annotation(pkl_path)
        self._labels = load_chad_labels(npy_path)
        self.num_frames = len(self._labels)

        # Reset buffers
        self._person_buffers = {}
        self._person_last_frame = {}
        self.score_timeline = []
        self.gt_timeline = []

    def process_frame(self, frame_idx: int) -> FrameAnomalyResult:
        """
        Process a single frame: update person buffers, run inference if ready.

        Args:
            frame_idx: The frame number to process.

        Returns:
            FrameAnomalyResult with anomaly scores for this frame.
        """
        result = FrameAnomalyResult(frame_idx=frame_idx)

        # Ground truth label
        if self._labels is not None and frame_idx < len(self._labels):
            result.gt_label = int(self._labels[frame_idx])

        # Get persons in this frame
        persons = self._annotation.get(frame_idx, {})
        result.num_persons = len(persons)

        # Update per-person skeleton buffers
        for person_id, (bbox, keypoints) in persons.items():
            pid = int(person_id)
            kp = np.array(keypoints, dtype=np.float32)

            if kp.size == 51:
                kp = kp.reshape(17, 3)
            elif kp.size == 34:
                kp = kp.reshape(17, 2)
            else:
                continue

            xy = kp[:, :2].copy()

            # Normalize relative to bounding box
            bbox_arr = np.array(bbox, dtype=np.float32)
            bx, by, bw, bh = bbox_arr[:4]
            if bw > 0 and bh > 0:
                cx, cy = bx + bw / 2, by + bh / 2
                xy[:, 0] = (xy[:, 0] - cx) / bw
                xy[:, 1] = (xy[:, 1] - cy) / bh

            pose = xy.flatten()  # (34,)

            # Check for temporal gap — reset buffer if too large
            if pid in self._person_last_frame:
                gap = frame_idx - self._person_last_frame[pid]
                if gap > MAX_FRAME_GAP:
                    self._person_buffers[pid] = []

            if pid not in self._person_buffers:
                self._person_buffers[pid] = []

            self._person_buffers[pid].append(pose)
            self._person_last_frame[pid] = frame_idx

            # Keep buffer at max needed size
            if len(self._person_buffers[pid]) > self.total_window:
                self._person_buffers[pid] = self._person_buffers[pid][-self.total_window:]

        # Run inference on all persons with full buffers
        ready_windows = []
        for pid, buf in self._person_buffers.items():
            if len(buf) >= self.total_window:
                window = np.stack(buf[-self.total_window:], axis=0)  # (18, 34)
                window = add_velocity_features(window)  # (18, 68)
                ready_windows.append(window)

        if ready_windows:
            batch = np.stack(ready_windows, axis=0)  # (N, 18, 68)
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

            x = batch_tensor[:, :self.seq_len, :]        # (N, 12, 68)
            target = batch_tensor[:, self.seq_len:, :]    # (N, 6, 68)

            with torch.no_grad():
                scores = self.model.anomaly_score(x, target)

            score_list = scores.cpu().numpy().tolist()
            result.scores = score_list
            result.max_score = max(score_list)
            result.mean_score = float(np.mean(score_list))
            result.is_anomaly = result.max_score > self.threshold

        self.score_timeline.append(result.max_score)
        self.gt_timeline.append(result.gt_label)

        return result

    def get_available_videos(self) -> list[str]:
        """List CHAD videos that have both annotations and labels."""
        ann_dir = self.data_root / "annotations"
        label_dir = self.data_root / "anomaly_labels"
        videos = []
        for pkl in sorted(ann_dir.glob("*.pkl")):
            stem = pkl.stem
            if (label_dir / f"{stem}.npy").exists():
                videos.append(stem)
        return videos

    def get_video_path(self, video_stem: str) -> Path | None:
        """Find the .mp4 video file for a given CHAD video stem."""
        video_dir = self.data_root.parent / "CHAD_Videos"
        mp4_path = video_dir / f"{video_stem}.mp4"
        if mp4_path.exists():
            return mp4_path
        return None
