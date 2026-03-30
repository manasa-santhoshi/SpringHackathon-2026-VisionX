"""
Metrics computation from YOLO detections.

All metrics are derived from model detections (not ground truth).
Uses homography to map pixel detections to ground coordinates for spatial reasoning.
Pedestrian metrics (person count, PSI) operate in pixel space and need no homography.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.detection.base import FrameDetections
from src.pipeline.homography import pixel_to_ground

# --- Person / vehicle classification for serialized detections ---
_PERSON_NAMES = {"pedestrian", "people", "person"}
_VEHICLE_NAMES = {"car", "van", "truck", "bus"}


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


class _ParkingSpaceLookup:
    """Precomputed numpy arrays for fast point-in-rect parking space lookup."""

    def __init__(self, parking_spaces: pd.DataFrame):
        corners = parking_spaces.iloc[:, 2:10].to_numpy(dtype=np.float64)
        xs = corners[:, [0, 2, 4, 6]]
        ys = corners[:, [1, 3, 5, 7]]
        self.min_x = xs.min(axis=1)
        self.max_x = xs.max(axis=1)
        self.min_y = ys.min(axis=1)
        self.max_y = ys.max(axis=1)
        self.areas = parking_spaces["area"].to_numpy()
        self.ids = parking_spaces["id"].to_numpy()

    def find_space(self, gx: float, gy: float) -> str | None:
        """Return the area label if (gx, gy) falls inside a parking space."""
        mask = (gx >= self.min_x) & (gx <= self.max_x) & (gy >= self.min_y) & (gy <= self.max_y)
        idx = np.argmax(mask)
        if mask[idx]:
            return str(self.areas[idx])
        return None

    def find_space_id(self, gx: float, gy: float) -> tuple[str | None, str | None]:
        """Return (space_id, area) if point is in a space, else (None, None)."""
        mask = (gx >= self.min_x) & (gx <= self.max_x) & (gy >= self.min_y) & (gy <= self.max_y)
        idx = np.argmax(mask)
        if mask[idx]:
            return self.ids[idx], str(self.areas[idx])
        return None, None


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
    lookup = _ParkingSpaceLookup(parking_spaces)

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
            space_id, area = lookup.find_space_id(gx, gy)
            if space_id is not None and space_id not in occupied_spaces:
                occupied_spaces.add(space_id)
                area_occupied[area] += 1

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
    lookup = _ParkingSpaceLookup(parking_spaces)

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
            area = lookup.find_space(gx, gy)
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


# =====================================================================
# Pedestrian metrics (pixel-space, no homography needed)
# =====================================================================


def compute_person_count(detections: list[FrameDetections]) -> dict:
    """
    Count unique persons across all frames using track IDs.

    Same structure as compute_vehicle_count but using frame.persons.
    """
    all_tracks = set()
    per_frame_counts = []

    for frame in detections:
        frame_track_ids = set()
        for p in frame.persons:
            all_tracks.add(p.track_id)
            frame_track_ids.add(p.track_id)
        per_frame_counts.append({
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "count": len(frame_track_ids),
        })

    return {
        "total_unique": len(all_tracks),
        "per_frame_counts": per_frame_counts,
    }


# --- Grid helpers for PSI ---

def _infer_frame_size(detections_data: list[dict]) -> tuple[int, int]:
    """Estimate frame width/height from max bbox coordinates."""
    max_x, max_y = 0.0, 0.0
    for frame in detections_data[:500]:
        for det in frame.get("vehicles", []) + frame.get("persons", []):
            cp = det.get("center_px")
            if cp:
                max_x = max(max_x, cp[0])
                max_y = max(max_y, cp[1])
    w = int(np.ceil(max_x / 100) * 100) if max_x > 0 else 3840
    h = int(np.ceil(max_y / 100) * 100) if max_y > 0 else 2160
    return w, h


def _make_grid(
    frame_w: int, frame_h: int, cols: int, rows: int,
) -> dict[str, tuple[float, float, float, float]]:
    """Return {cell_name: (x0, y0, x1, y1)} for an NxM pixel grid."""
    cw = frame_w / cols
    ch = frame_h / rows
    return {
        f"cell_{r}_{c}": (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
        for r in range(rows)
        for c in range(cols)
    }


def _assign_cell(
    cx: float, cy: float, grid: dict[str, tuple[float, float, float, float]],
) -> str:
    for name, (x0, y0, x1, y1) in grid.items():
        if x0 <= cx < x1 and y0 <= cy < y1:
            return name
    return "outside"


def compute_psi(
    detections_data: list[dict],
    grid_cols: int = 4,
    grid_rows: int = 4,
) -> dict:
    """
    Compute Parking Stress Index per grid zone.

    PSI = 0.4 * norm(passengers) + 0.4 * norm(vehicles) + 0.2 * norm(ratio)
    Scaled 0-10.

    Args:
        detections_data: Serialized detection dicts (with 'vehicles' and 'persons' keys).
        grid_cols: Number of grid columns.
        grid_rows: Number of grid rows.

    Returns:
        Dict with grid config and per-zone PSI summary.
    """
    fw, fh = _infer_frame_size(detections_data)
    grid = _make_grid(fw, fh, grid_cols, grid_rows)

    # Accumulate per-zone per-frame counts
    zone_peds: dict[str, list[int]] = {name: [] for name in grid}
    zone_vehs: dict[str, list[int]] = {name: [] for name in grid}

    for frame in detections_data:
        frame_peds: dict[str, int] = defaultdict(int)
        frame_vehs: dict[str, int] = defaultdict(int)

        for det in frame.get("persons", []):
            cp = det.get("center_px")
            if cp:
                cell = _assign_cell(cp[0], cp[1], grid)
                if cell != "outside":
                    frame_peds[cell] += 1

        for det in frame.get("vehicles", []):
            cp = det.get("center_px")
            if cp:
                cell = _assign_cell(cp[0], cp[1], grid)
                if cell != "outside":
                    frame_vehs[cell] += 1

        for name in grid:
            zone_peds[name].append(frame_peds.get(name, 0))
            zone_vehs[name].append(frame_vehs.get(name, 0))

    # Compute per-zone averages
    zones = []
    avg_peds_all = []
    avg_vehs_all = []
    avg_ratio_all = []

    for name in grid:
        peds_arr = np.array(zone_peds[name], dtype=float)
        vehs_arr = np.array(zone_vehs[name], dtype=float)
        avg_p = float(peds_arr.mean())
        avg_v = float(vehs_arr.mean())
        avg_r = float(np.mean(peds_arr / np.where(vehs_arr > 0, vehs_arr, 1.0)))

        avg_peds_all.append(avg_p)
        avg_vehs_all.append(avg_v)
        avg_ratio_all.append(avg_r)

        zones.append({
            "area": name,
            "avg_peds": round(avg_p, 2),
            "peak_peds": int(peds_arr.max()),
            "avg_vehicles": round(avg_v, 2),
            "peak_vehicles": int(vehs_arr.max()),
        })

    # Normalize and compute PSI
    def _norm(vals: list[float]) -> np.ndarray:
        arr = np.array(vals)
        mn, mx = arr.min(), arr.max()
        return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)

    norm_p = _norm(avg_peds_all)
    norm_v = _norm(avg_vehs_all)
    norm_r = _norm(avg_ratio_all)
    psi_scores = (0.4 * norm_p + 0.4 * norm_v + 0.2 * norm_r) * 10

    for i, zone in enumerate(zones):
        zone["avg_psi"] = round(float(psi_scores[i]), 2)
        # Peak PSI: compute per-frame PSI for this zone and take max
        peds_arr = np.array(zone_peds[zone["area"]], dtype=float)
        vehs_arr = np.array(zone_vehs[zone["area"]], dtype=float)
        ratio_arr = peds_arr / np.where(vehs_arr > 0, vehs_arr, 1.0)

        def _norm_arr(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)

        frame_psi = (
            0.4 * _norm_arr(peds_arr)
            + 0.4 * _norm_arr(vehs_arr)
            + 0.2 * _norm_arr(ratio_arr)
        ) * 10
        zone["peak_psi"] = round(float(frame_psi.max()), 2)

    return {
        "grid": {
            "cols": grid_cols,
            "rows": grid_rows,
            "frame_width": fw,
            "frame_height": fh,
        },
        "zones": zones,
    }
