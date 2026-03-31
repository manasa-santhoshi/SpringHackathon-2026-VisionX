"""
Real-time streaming pipeline for live video inference in Streamlit.

Provides incremental metric accumulators that update frame-by-frame,
and a frame annotator for drawing detection overlays.

Usage:
    from src.pipeline.realtime import MetricsAccumulator, draw_detections
"""

from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from src.detection.base import DetectedVehicle, FrameDetections
from src.pipeline.homography import pixel_to_ground
from src.pipeline.metrics import _ParkingSpaceLookup, _find_parking_space, _point_in_rect

# Entrance zone bounds (from parking_map.yml WAYPOINTS.EXT area)
ENTRANCE_X_MIN, ENTRANCE_X_MAX = 0.0, 30.0
ENTRANCE_Y_MIN, ENTRANCE_Y_MAX = 65.0, 80.0

# Track is considered lost after this many frames without appearing
TRACK_TIMEOUT_FRAMES = 30


def _near_entrance(gx: float, gy: float) -> bool:
    return (ENTRANCE_X_MIN <= gx <= ENTRANCE_X_MAX
            and ENTRANCE_Y_MIN <= gy <= ENTRANCE_Y_MAX)


class MetricsAccumulator:
    """
    Incrementally computes all 4 parking metrics frame-by-frame.

    Call add_frame() for each new detection, then get_snapshot() for current state.
    """

    def __init__(
        self,
        H: np.ndarray,
        parking_spaces: pd.DataFrame,
        sample_interval: float = 1.0,
    ):
        self.H = H
        self.parking_spaces = parking_spaces
        self.sample_interval = sample_interval
        self.total_spaces = len(parking_spaces)
        self.area_totals = parking_spaces.groupby("area").size().to_dict()
        self._lookup = _ParkingSpaceLookup(parking_spaces)

        # Vehicle count state
        self.all_tracks: set[int] = set()
        self.track_class_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.per_frame_counts: list[dict] = []

        # Occupancy state
        self.last_sampled_time = -sample_interval
        self.occ_timestamps: list[float] = []
        self.occ_occupied: list[int] = []
        self.occ_free: list[int] = []
        self.current_occupied_space_ids: set[int] = set()

        # Dwell time state
        self.active_tracks: dict[int, list[tuple[float, float, float]]] = {}
        self.track_last_seen: dict[int, int] = {}  # track_id -> last frame_idx
        self.completed_dwells: list[dict] = []
        self.first_timestamp: float | None = None
        self.current_frame_idx = 0

        # Entry/exit state
        self.track_first_pos: dict[int, tuple[float, float, float]] = {}
        self.track_last_pos: dict[int, tuple[float, float, float]] = {}
        self.track_class: dict[int, str] = {}
        self.entries: list[dict] = []
        self.exits: list[dict] = []
        self.entry_track_ids: set[int] = set()
        self.exit_track_ids: set[int] = set()

    def add_frame(self, frame: FrameDetections) -> None:
        """Process a single frame's detections and update all metrics."""
        if self.first_timestamp is None:
            self.first_timestamp = frame.timestamp
        self.current_frame_idx = frame.frame_idx

        current_track_ids = set()

        # --- Per-vehicle processing ---
        for v in frame.vehicles:
            current_track_ids.add(v.track_id)

            # Vehicle count
            self.all_tracks.add(v.track_id)
            self.track_class_votes[v.track_id][v.class_name] += 1

            # Ground coords
            gx, gy = pixel_to_ground(self.H, v.center_px[0], v.center_px[1])

            # Dwell tracking
            if v.track_id not in self.active_tracks:
                self.active_tracks[v.track_id] = []
            self.active_tracks[v.track_id].append((frame.timestamp, gx, gy))
            self.track_last_seen[v.track_id] = frame.frame_idx

            # Entry/exit tracking
            if v.track_id not in self.track_first_pos:
                self.track_first_pos[v.track_id] = (frame.timestamp, gx, gy)
                self.track_class[v.track_id] = v.class_name
                # Check for entry
                if _near_entrance(gx, gy) and v.track_id not in self.entry_track_ids:
                    self.entries.append({
                        "track_id": v.track_id,
                        "timestamp": round(frame.timestamp, 2),
                        "class": v.class_name,
                    })
                    self.entry_track_ids.add(v.track_id)
            self.track_last_pos[v.track_id] = (frame.timestamp, gx, gy)

        # Per-frame count
        self.per_frame_counts.append({
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "count": len(current_track_ids),
        })

        # --- Occupancy sampling ---
        if frame.timestamp - self.last_sampled_time >= self.sample_interval:
            self.last_sampled_time = frame.timestamp
            occupied_spaces = set()

            for v in frame.vehicles:
                gx, gy = pixel_to_ground(self.H, v.center_px[0], v.center_px[1])
                space_id, _ = self._lookup.find_space_id(gx, gy)
                if space_id is not None:
                    occupied_spaces.add(space_id)

            self.current_occupied_space_ids = {int(sid) for sid in occupied_spaces}
            n_occupied = len(occupied_spaces)
            self.occ_timestamps.append(round(frame.timestamp, 2))
            self.occ_occupied.append(n_occupied)
            self.occ_free.append(self.total_spaces - n_occupied)

        # --- Finalize lost tracks (dwell + exit) ---
        lost_tracks = [
            tid for tid, last_frame in self.track_last_seen.items()
            if frame.frame_idx - last_frame > TRACK_TIMEOUT_FRAMES
            and tid in self.active_tracks
        ]

        for tid in lost_tracks:
            self._finalize_track(tid)

    def _majority_class_counts(self) -> dict[str, int]:
        """Count tracks by majority class (most frequent classification)."""
        by_class: dict[str, int] = defaultdict(int)
        for votes in self.track_class_votes.values():
            majority_class = max(votes, key=votes.get)
            by_class[majority_class] += 1
        return dict(by_class)

    def _finalize_track(self, track_id: int) -> None:
        """Finalize dwell times and exit events for a lost track."""
        timeline = self.active_tracks.pop(track_id, [])
        if not timeline:
            return

        # Dwell time computation with gap tolerance
        gap_tolerance = 3.0
        parked_start = None
        parked_area = None
        gap_start = None

        for ts, gx, gy in timeline:
            area = self._lookup.find_space(gx, gy)
            if area is not None:
                if parked_start is None:
                    parked_start = ts
                    parked_area = area
                gap_start = None
            else:
                if parked_start is not None:
                    if gap_start is None:
                        gap_start = ts
                    elif ts - gap_start > gap_tolerance:
                        duration = gap_start - parked_start
                        if duration > 2.0:
                            self.completed_dwells.append({
                                "track_id": track_id,
                                "duration_sec": round(duration, 2),
                                "area": parked_area,
                            })
                        parked_start = None
                        parked_area = None
                        gap_start = None

        # Handle still-parked at track loss
        if parked_start is not None:
            end_ts = gap_start if gap_start is not None else timeline[-1][0]
            duration = end_ts - parked_start
            if duration > 2.0:
                self.completed_dwells.append({
                    "track_id": track_id,
                    "duration_sec": round(duration, 2),
                    "area": parked_area,
                })

        # Exit detection
        if track_id in self.track_last_pos and track_id not in self.exit_track_ids:
            _, gx, gy = self.track_last_pos[track_id]
            if _near_entrance(gx, gy):
                self.exits.append({
                    "track_id": track_id,
                    "timestamp": round(self.track_last_pos[track_id][0], 2),
                    "class": self.track_class.get(track_id, "unknown"),
                })
                self.exit_track_ids.add(track_id)

        # Cleanup
        self.track_last_seen.pop(track_id, None)

    def get_snapshot(self) -> dict:
        """Return current state of all metrics for dashboard rendering."""
        # Compute active dwell times for vehicles currently parked
        active_parked = 0
        active_dwells: list[dict] = []
        for tid, timeline in self.active_tracks.items():
            if not timeline:
                continue
            last_ts, gx, gy = timeline[-1]
            if self._lookup.find_space(gx, gy) is None:
                continue
            active_parked += 1
            # Find when this vehicle first entered a parking space (with gap tolerance)
            gap_tolerance = 3.0
            parked_start = None
            gap_start = None
            for ts, tx, ty in timeline:
                area = self._lookup.find_space(tx, ty)
                if area is not None:
                    if parked_start is None:
                        parked_start = ts
                    gap_start = None
                else:
                    if parked_start is not None:
                        if gap_start is None:
                            gap_start = ts
                        elif ts - gap_start > gap_tolerance:
                            parked_start = None
                            gap_start = None
            if parked_start is not None:
                duration = last_ts - parked_start
                if duration > 2.0:
                    active_dwells.append({
                        "track_id": tid,
                        "duration_sec": round(duration, 2),
                    })

        # Merge completed + active dwells for stats
        all_dwells = [d.copy() for d in self.completed_dwells] + active_dwells
        all_durations = [d["duration_sec"] for d in all_dwells]

        return {
            "vehicle_count": {
                "total_unique": len(self.all_tracks),
                "by_class": self._majority_class_counts(),
                "current_frame_count": self.per_frame_counts[-1]["count"] if self.per_frame_counts else 0,
            },
            "occupancy": {
                "timestamps": self.occ_timestamps,
                "occupied": self.occ_occupied,
                "free": self.occ_free,
                "total_spaces": self.total_spaces,
                "current_occupied": self.occ_occupied[-1] if self.occ_occupied else 0,
                "occupied_space_ids": list(self.current_occupied_space_ids),
            },
            "dwell": {
                "completed": len(self.completed_dwells),
                "active_parked": active_parked,
                "stats": {
                    "mean_sec": round(float(np.mean(all_durations)), 2),
                    "median_sec": round(float(np.median(all_durations)), 2),
                    "count": len(all_durations),
                } if all_durations else {},
                "dwell_times": all_dwells,
            },
            "entry_exit": {
                "entry_count": len(self.entries),
                "exit_count": len(self.exits),
            },
        }


def draw_detections(
    frame_bgr: np.ndarray,
    detections: FrameDetections,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes with track IDs on a video frame.

    Args:
        frame_bgr: BGR frame from OpenCV.
        detections: Frame detections to draw.
        scale: Resize scale for display (0.5 = half size).

    Returns:
        Annotated frame in RGB format for st.image().
    """
    frame = frame_bgr.copy()

    for v in detections.vehicles:
        x1, y1, x2, y2 = [int(c) for c in v.bbox]
        color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{v.track_id} {v.class_name} {v.confidence:.1%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Resize for display
    if scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # BGR → RGB for Streamlit
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_detections_from_result(
    result,
    frame_idx: int,
    timestamp: float,
    vehicle_classes: dict[int, str],
) -> FrameDetections:
    """
    Extract FrameDetections from a single YOLO tracking result.

    Args:
        result: Single YOLO result object.
        frame_idx: Current frame index.
        timestamp: Current timestamp in seconds.
        vehicle_classes: Mapping of class IDs to names.

    Returns:
        FrameDetections for this frame.
    """
    vehicles = []
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in vehicle_classes:
                continue

            track_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            vehicles.append(
                DetectedVehicle(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=vehicle_classes[cls_id],
                    center_px=(cx, cy),
                )
            )

    return FrameDetections(
        frame_idx=frame_idx,
        timestamp=timestamp,
        vehicles=vehicles,
    )
