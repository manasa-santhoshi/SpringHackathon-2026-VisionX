"""
Abstract base class for parking lot vehicle detectors.

Defines the interface that any detector (YOLO, ground-truth passthrough, etc.) must implement.
Inherits from torch.nn.Module to satisfy the PyTorch framework requirement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class DetectedVehicle:
    """A single detected vehicle in a frame."""
    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    center_px: tuple[float, float]  # (cx, cy) pixel center


@dataclass
class FrameDetections:
    """All detections for a single video frame."""
    frame_idx: int
    timestamp: float  # seconds from video start
    vehicles: list[DetectedVehicle] = field(default_factory=list)
    persons: list[DetectedVehicle] = field(default_factory=list)


class ParkingDetector(nn.Module, ABC):
    """
    Abstract base detector for parking lot vehicle detection and tracking.

    Subclasses implement detect_and_track() which processes a video and returns
    per-frame detections with persistent track IDs.
    """

    @abstractmethod
    def detect_and_track(self, video_path: str, **kwargs: Any) -> list[FrameDetections]:
        """
        Process a video file and return per-frame vehicle detections with tracking.

        Args:
            video_path: Path to the video file.

        Returns:
            List of FrameDetections, one per processed frame.
        """
        ...
