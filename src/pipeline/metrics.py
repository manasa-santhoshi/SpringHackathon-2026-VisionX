"""
Metrics computation from YOLO detections.

All metrics are derived from model detections (not ground truth).
Uses homography to map pixel detections to ground coordinates for spatial reasoning.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.detection.base import FrameDetections
from src.pipeline.homography import pixel_to_ground


def _point_in_rect(
    px: float, py: float, corners: np.ndarray
) -> bool:
    """
    Check if point (px, py) is inside an axis-aligned rectangle defined by 4 corners.

    corners: [top_left_x, top_left_y, top_right_x, top_right_y,
              btm_right_x, btm_right_y, btm_left_x, btm_left_y]
    """
    xs = [corners[0], corners[2], corners[4], corners[6]]
    ys = [corners[1], corners[3], corners[5], corners[7]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x <= px <= max_x and min_y <= py <= max_y


def _find_parking_space(
    gx: float, gy: float, parking_spaces: pd.DataFrame
) -> str | None:
    """Return the area label if (gx, gy) falls inside a parking space, else None."""
    for _, row in parking_spaces.iterrows():
        corners = row.iloc[2:10].to_numpy()
        if _point_in_rect(gx, gy, corners):
            return row["area"]
    return None


def compute_vehicle_count(detections: list[FrameDetections]) -> dict:
    """
    Count unique vehicles across all frames using track IDs.

    Returns dict with total_unique, by_class breakdown, and per_frame_counts.
    """
    all_tracks = set()
    tracks_by_class = defaultdict(set)
    per_frame_counts = []

    for frame in detections:
        frame_track_ids = set()
        for v in frame.vehicles:
            all_tracks.add(v.track_id)
            tracks_by_class[v.class_name].add(v.track_id)
            frame_track_ids.add(v.track_id)
        per_frame_counts.append({
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "count": len(frame_track_ids),
        })

    return {
        "total_unique": len(all_tracks),
        "by_class": {cls: len(ids) for cls, ids in tracks_by_class.items()},
        "per_frame_counts": per_frame_counts,
    }


def compute_occupancy_timeline(
    detections: list[FrameDetections],
    H: np.ndarray,
    parking_spaces: pd.DataFrame,
    sample_interval: float = 1.0,
) -> dict:
    """
    Compute parking occupancy over time by mapping detections to parking spaces.

    Samples frames at the given interval. For each sampled frame, transforms detection
    pixel centers to ground coords and checks which parking spaces are occupied.

    Returns dict with timestamps, occupied/free counts, total_spaces, and by_area breakdown.
    """
    total_spaces = len(parking_spaces)
    # Precompute per-area space counts
    area_totals = parking_spaces.groupby("area").size().to_dict()

    timestamps = []
    occupied_counts = []
    free_counts = []
    by_area = {area: {"occupied": [], "total": count} for area, count in area_totals.items()}

    last_sampled_time = -sample_interval  # ensure first frame is sampled

    for frame in detections:
        if frame.timestamp - last_sampled_time < sample_interval:
            continue
        last_sampled_time = frame.timestamp

        # Track which spaces are occupied this frame
        occupied_spaces = set()
        area_occupied = defaultdict(int)

        for v in frame.vehicles:
            gx, gy = pixel_to_ground(H, v.center_px[0], v.center_px[1])
            for _, row in parking_spaces.iterrows():
                corners = row.iloc[2:10].to_numpy()
                if _point_in_rect(gx, gy, corners):
                    space_id = row["id"]
                    if space_id not in occupied_spaces:
                        occupied_spaces.add(space_id)
                        area_occupied[row["area"]] += 1
                    break  # vehicle can only be in one space

        n_occupied = len(occupied_spaces)
        timestamps.append(round(frame.timestamp, 2))
        occupied_counts.append(n_occupied)
        free_counts.append(total_spaces - n_occupied)

        for area in by_area:
            by_area[area]["occupied"].append(area_occupied.get(area, 0))

    return {
        "timestamps": timestamps,
        "occupied": occupied_counts,
        "free": free_counts,
        "total_spaces": total_spaces,
        "by_area": by_area,
    }


def compute_dwell_times(
    detections: list[FrameDetections],
    H: np.ndarray,
    parking_spaces: pd.DataFrame,
) -> dict:
    """
    Compute how long each tracked vehicle stays parked.

    For each track, walks through its frame appearances, checks if the vehicle is inside
    a parking space, and measures contiguous parked durations.

    Returns dict with individual dwell_times and aggregate stats.
    """
    # Build per-track timeline: list of (timestamp, ground_x, ground_y)
    track_timeline: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    for frame in detections:
        for v in frame.vehicles:
            gx, gy = pixel_to_ground(H, v.center_px[0], v.center_px[1])
            track_timeline[v.track_id].append((frame.timestamp, gx, gy))

    dwell_times = []
    first_timestamp = detections[0].timestamp if detections else 0.0
    last_timestamp = detections[-1].timestamp if detections else 0.0

    for track_id, timeline in track_timeline.items():
        timeline.sort(key=lambda x: x[0])

        # Find contiguous parked segments
        parked_start = None
        parked_area = None

        for ts, gx, gy in timeline:
            area = _find_parking_space(gx, gy, parking_spaces)
            if area is not None:
                if parked_start is None:
                    parked_start = ts
                    parked_area = area
            else:
                if parked_start is not None:
                    duration = ts - parked_start
                    if duration > 2.0:  # minimum 2 seconds to count
                        censored_start = abs(parked_start - first_timestamp) < 1.0
                        dwell_times.append({
                            "track_id": track_id,
                            "duration_sec": round(duration, 2),
                            "area": parked_area,
                            "censored": censored_start,
                        })
                    parked_start = None
                    parked_area = None

        # Handle still-parked at end of video
        if parked_start is not None:
            duration = timeline[-1][0] - parked_start
            if duration > 2.0:
                censored_end = abs(timeline[-1][0] - last_timestamp) < 1.0
                censored_start = abs(parked_start - first_timestamp) < 1.0
                dwell_times.append({
                    "track_id": track_id,
                    "duration_sec": round(duration, 2),
                    "area": parked_area,
                    "censored": censored_start or censored_end,
                })

    # Compute stats
    durations = [d["duration_sec"] for d in dwell_times]
    stats = {}
    if durations:
        stats = {
            "mean_sec": round(float(np.mean(durations)), 2),
            "median_sec": round(float(np.median(durations)), 2),
            "min_sec": round(min(durations), 2),
            "max_sec": round(max(durations), 2),
            "count": len(durations),
            "censored_count": sum(1 for d in dwell_times if d["censored"]),
        }

    return {"dwell_times": dwell_times, "stats": stats}


def compute_entry_exit(
    detections: list[FrameDetections],
    H: np.ndarray,
) -> dict:
    """
    Count vehicles entering and exiting the parking lot.

    Uses the entrance zone from parking_map.yml: approximately [5,70] to [25,80] in local coords.
    A vehicle entering has its first tracked appearance near the entrance.
    A vehicle exiting has its last tracked appearance near the entrance.

    Returns dict with entry/exit events, counts, and cumulative timeline.
    """
    # Entrance zone bounds (from parking_map.yml WAYPOINTS.EXT area, with margin)
    ENTRANCE_X_MIN, ENTRANCE_X_MAX = 0.0, 30.0
    ENTRANCE_Y_MIN, ENTRANCE_Y_MAX = 65.0, 80.0

    def near_entrance(gx: float, gy: float) -> bool:
        return (ENTRANCE_X_MIN <= gx <= ENTRANCE_X_MAX
                and ENTRANCE_Y_MIN <= gy <= ENTRANCE_Y_MAX)

    # Build per-track first/last appearances
    track_first: dict[int, tuple[float, float, float]] = {}  # track_id -> (ts, gx, gy)
    track_last: dict[int, tuple[float, float, float]] = {}
    track_class: dict[int, str] = {}

    for frame in detections:
        for v in frame.vehicles:
            gx, gy = pixel_to_ground(H, v.center_px[0], v.center_px[1])
            if v.track_id not in track_first:
                track_first[v.track_id] = (frame.timestamp, gx, gy)
                track_class[v.track_id] = v.class_name
            track_last[v.track_id] = (frame.timestamp, gx, gy)

    entries = []
    exits = []

    for track_id in track_first:
        first_ts, first_gx, first_gy = track_first[track_id]
        last_ts, last_gx, last_gy = track_last[track_id]

        if near_entrance(first_gx, first_gy):
            entries.append({
                "track_id": track_id,
                "timestamp": round(first_ts, 2),
                "class": track_class[track_id],
            })

        if near_entrance(last_gx, last_gy):
            exits.append({
                "track_id": track_id,
                "timestamp": round(last_ts, 2),
                "class": track_class[track_id],
            })

    entries.sort(key=lambda x: x["timestamp"])
    exits.sort(key=lambda x: x["timestamp"])

    # Build cumulative timeline at regular intervals
    if detections:
        max_time = detections[-1].timestamp
        interval = 30.0  # 30-second bins
        time_bins = list(np.arange(0, max_time + interval, interval))
        cum_entries = []
        cum_exits = []
        for t in time_bins:
            cum_entries.append(sum(1 for e in entries if e["timestamp"] <= t))
            cum_exits.append(sum(1 for e in exits if e["timestamp"] <= t))
        timeline = {
            "timestamps": [round(t, 1) for t in time_bins],
            "cumulative_entries": cum_entries,
            "cumulative_exits": cum_exits,
        }
    else:
        timeline = {"timestamps": [], "cumulative_entries": [], "cumulative_exits": []}

    return {
        "entries": entries,
        "exits": exits,
        "entry_count": len(entries),
        "exit_count": len(exits),
        "timeline": timeline,
    }
