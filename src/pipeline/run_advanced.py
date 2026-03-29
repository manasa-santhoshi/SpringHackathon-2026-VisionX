"""
src/pipeline/run_advanced_from_detections.py

Replicates run_advanced.py metrics entirely from detections.json:
  - Parking Stress Index (PSI)
  - Pedestrian Flow Direction (vx/vy per grid cell)
  - Dead Zone Detection
  - Pedestrian Surge Predictor

No DLP SDK needed. Works from your existing detections.json.
Auto-grid splits the frame into NxM cells as zones.

Run:
    python -m src.pipeline.run_advanced_from_detections --scene DJI_0012
    python -m src.pipeline.run_advanced_from_detections --scene DJI_0012 --grid 4 4
    python -m src.pipeline.run_advanced_from_detections --scene DJI_0012 --grid 6 6 --fps 30
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PERSON_NAMES  = {"pedestrian", "people", "person"}
VEHICLE_NAMES = {"car", "van", "truck", "bus"}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_detections(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    frames = []
    for fid, val in data.items():
        entry = val if isinstance(val, dict) else {"vehicles": val}
        entry["frame_idx"] = int(fid)
        frames.append(entry)
    return sorted(frames, key=lambda x: x["frame_idx"])


def get_dets(frame: dict) -> list:
    return frame.get("vehicles", frame.get("detections", []))


def is_person(det):
    return det.get("class_name", "").lower() in PERSON_NAMES


def is_vehicle(det):
    return det.get("class_name", "").lower() in VEHICLE_NAMES


def center(det):
    if "center_px" in det:
        return det["center_px"]
    b = det.get("bbox", [])
    if len(b) == 4:
        return [(b[0]+b[2])/2, (b[1]+b[3])/2]
    return None


def infer_frame_size(frames: list) -> tuple:
    """Estimate frame width/height from max bbox coordinates seen."""
    max_x, max_y = 0, 0
    for frame in frames[:500]:  # sample first 500 frames
        for det in get_dets(frame):
            c = center(det)
            if c:
                max_x = max(max_x, c[0])
                max_y = max(max_y, c[1])
    # round up to nearest 100
    w = int(np.ceil(max_x / 100) * 100) if max_x > 0 else 3840
    h = int(np.ceil(max_y / 100) * 100) if max_y > 0 else 2160
    return w, h


def make_grid(frame_w: int, frame_h: int, cols: int, rows: int) -> dict:
    """
    Returns dict of cell_name -> (x0, y0, x1, y1).
    """
    cw = frame_w / cols
    ch = frame_h / rows
    cells = {}
    for r in range(rows):
        for c in range(cols):
            name = f"cell_{r}_{c}"
            cells[name] = (c * cw, r * ch, (c+1) * cw, (r+1) * ch)
    return cells


def assign_cell(cx, cy, grid: dict) -> str:
    for name, (x0, y0, x1, y1) in grid.items():
        if x0 <= cx < x1 and y0 <= cy < y1:
            return name
    return "outside"


# ─────────────────────────────────────────────────────────────
# STEP 1 — Build per-frame, per-cell summary DataFrame
# ─────────────────────────────────────────────────────────────

def build_frame_df(frames: list, grid: dict, fps: float) -> pd.DataFrame:
    """
    For each frame x cell: count passengers, vehicles, compute velocity.
    Velocity = displacement of same track_id between consecutive frames.
    """
    # First pass: build track position history {track_id: [(frame_idx, cx, cy)]}
    track_history = defaultdict(list)
    for frame in frames:
        fidx = frame.get("frame_idx", frame.get("frame_id", 0))
        for det in get_dets(frame):
            if not is_person(det):
                continue
            tid = det.get("track_id")
            c = center(det)
            if tid is not None and c:
                track_history[tid].append((fidx, c[0], c[1]))

    # Compute instantaneous velocity per (track_id, frame_idx)
    vel = {}  # (track_id, frame_idx) -> (vx, vy) in px/s
    for tid, positions in track_history.items():
        positions.sort(key=lambda x: x[0])
        for i in range(1, len(positions)):
            f0, x0, y0 = positions[i-1]
            f1, x1, y1 = positions[i]
            dt = (f1 - f0) / fps
            if dt > 0:
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                vel[(tid, f1)] = (vx, vy)

    # Second pass: build records
    records = []
    for frame in frames:
        fidx  = frame.get("frame_idx", frame.get("frame_id", 0))
        ts    = frame.get("timestamp", fidx / fps)
        dets  = get_dets(frame)

        # Accumulate per cell
        cell_peds    = defaultdict(int)
        cell_vehs    = defaultdict(int)
        cell_vx_sum  = defaultdict(float)
        cell_vy_sum  = defaultdict(float)
        cell_vx_cnt  = defaultdict(int)

        for det in dets:
            c = center(det)
            if not c:
                continue
            cell = assign_cell(c[0], c[1], grid)

            if is_person(det):
                cell_peds[cell] += 1
                tid = det.get("track_id")
                if tid and (tid, fidx) in vel:
                    vx, vy = vel[(tid, fidx)]
                    cell_vx_sum[cell] += vx
                    cell_vy_sum[cell] += vy
                    cell_vx_cnt[cell] += 1

            elif is_vehicle(det):
                cell_vehs[cell] += 1

        # Emit one row per cell (even zero-pedestrian cells)
        for cell in grid:
            n_peds = cell_peds[cell]
            n_vehs = cell_vehs[cell]
            cnt    = cell_vx_cnt[cell]
            records.append({
                "frame_idx":  fidx,
                "timestamp":  round(ts, 2),
                "area":       cell,
                "passengers": n_peds,
                "vehicles":   n_vehs,
                "ratio":      n_peds / n_vehs if n_vehs > 0 else 0.0,
                "avg_vx":     round(cell_vx_sum[cell] / cnt, 4) if cnt > 0 else 0.0,
                "avg_vy":     round(cell_vy_sum[cell] / cnt, 4) if cnt > 0 else 0.0,
                "n_moving":   cnt,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# METRIC 1 — Parking Stress Index
# ─────────────────────────────────────────────────────────────

def compute_psi(df: pd.DataFrame) -> pd.DataFrame:
    """
    PSI = 0.4 * norm(passengers) + 0.4 * norm(vehicles) + 0.2 * norm(ratio)
    Scaled 0–10. Matches run_advanced.py formula exactly.
    """
    df = df.copy()

    def norm(col):
        mn, mx = col.min(), col.max()
        return pd.Series(0.0, index=col.index) if mx == mn else (col - mn) / (mx - mn)

    df["psi"] = (
        0.4 * norm(df["passengers"]) +
        0.4 * norm(df["vehicles"])   +
        0.2 * norm(df["ratio"])
    ) * 10
    df["psi"] = df["psi"].round(2)
    return df


# ─────────────────────────────────────────────────────────────
# METRIC 2 — Flow Direction (already in df as avg_vx / avg_vy)
# ─────────────────────────────────────────────────────────────

def compute_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Add flow speed and direction angle to df."""
    df = df.copy()
    df["flow_speed"] = np.sqrt(df["avg_vx"]**2 + df["avg_vy"]**2).round(2)
    df["flow_angle"] = np.degrees(np.arctan2(df["avg_vy"], df["avg_vx"])).round(1)
    return df


# ─────────────────────────────────────────────────────────────
# METRIC 3 — Dead Zone Detection
# ─────────────────────────────────────────────────────────────

def compute_dead_zones(df: pd.DataFrame, min_consecutive: int = 10) -> pd.DataFrame:
    """
    Find areas with zero pedestrian activity for N+ consecutive frames.
    Matches run_advanced.py logic exactly.
    """
    records = []
    for area, group in df.groupby("area"):
        group = group.sort_values("timestamp")
        consecutive = 0
        start_ts = None

        for _, row in group.iterrows():
            if row["passengers"] == 0:
                if consecutive == 0:
                    start_ts = row["timestamp"]
                consecutive += 1
            else:
                if consecutive >= min_consecutive:
                    records.append({
                        "area":            area,
                        "start_time":      start_ts,
                        "end_time":        row["timestamp"],
                        "duration_frames": consecutive,
                    })
                consecutive = 0

        if consecutive >= min_consecutive:
            records.append({
                "area":            area,
                "start_time":      start_ts,
                "end_time":        group["timestamp"].iloc[-1],
                "duration_frames": consecutive,
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["area", "start_time", "end_time", "duration_frames"]
    )


# ─────────────────────────────────────────────────────────────
# METRIC 4 — Surge Predictor
# ─────────────────────────────────────────────────────────────

def compute_surge_predictor(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Rolling mean + std of passenger count per area.
    Surge flag = current count > mean + 2*std over last `window` frames.
    """
    records = []
    for area, group in df.groupby("area"):
        group = group.sort_values("timestamp").copy()
        group["roll_mean"] = group["passengers"].rolling(window, min_periods=1).mean()
        group["roll_std"]  = group["passengers"].rolling(window, min_periods=1).std().fillna(0)
        group["surge"]     = group["passengers"] > (group["roll_mean"] + 2 * group["roll_std"])
        group["area"]      = area
        records.append(group)

    return pd.concat(records).sort_values(["area", "timestamp"])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",  required=True, help="Scene name, e.g. DJI_0012")
    ap.add_argument("--grid",   nargs=2, type=int, default=[4, 4],
                    metavar=("COLS", "ROWS"),
                    help="Grid divisions (default: 4 4 = 16 zones)")
    ap.add_argument("--fps",    type=float, default=29.97,
                    help="Video FPS for velocity calc (default 29.97)")
    ap.add_argument("--min-dead", type=int, default=10,
                    help="Min consecutive zero-ped frames for dead zone (default 10)")
    ap.add_argument("--surge-window", type=int, default=30,
                    help="Rolling window frames for surge predictor (default 30)")
    args = ap.parse_args()

    det_path = PROCESSED_DIR / args.scene / "detections.json"
    out_dir  = PROCESSED_DIR / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  ADVANCED METRICS — {args.scene}")
    print(f"  Grid: {args.grid[0]} cols × {args.grid[1]} rows = "
          f"{args.grid[0]*args.grid[1]} zones")
    print(f"{'='*58}\n")

    # Load
    print("Loading detections...")
    frames = load_detections(det_path)
    print(f"  {len(frames):,} frames loaded")

    # Infer frame size
    fw, fh = infer_frame_size(frames)
    print(f"  Inferred frame size: {fw} × {fh} px")

    # Build grid
    grid = make_grid(fw, fh, args.grid[0], args.grid[1])
    print(f"  Grid zones: {list(grid.keys())[:4]} ...")

    # Build base DataFrame
    print("\nBuilding per-frame per-cell summary (computing velocities)...")
    df = build_frame_df(frames, grid, args.fps)
    print(f"  DataFrame shape: {df.shape}")

    # ── Metric 1: PSI ──
    print("\n── Parking Stress Index (PSI) ───────────────────────")
    df = compute_psi(df)
    psi_summary = df.groupby("area")["psi"].agg(["mean","max"]).round(2)
    psi_summary.columns = ["avg_psi", "peak_psi"]
    psi_summary = psi_summary.sort_values("peak_psi", ascending=False)
    print(psi_summary.to_string())
    psi_path = out_dir / "psi.csv"
    df[["timestamp","area","passengers","vehicles","ratio","psi"]].to_csv(psi_path, index=False)
    print(f"  Saved → {psi_path}")

    # ── Metric 2: Flow ──
    print("\n── Pedestrian Flow Direction ────────────────────────")
    df = compute_flow(df)
    flow_summary = df[df["n_moving"] > 0].groupby("area")[["avg_vx","avg_vy","flow_speed"]].mean().round(2)
    print(flow_summary.to_string())
    flow_path = out_dir / "flow.csv"
    df[["timestamp","area","avg_vx","avg_vy","flow_speed","flow_angle","n_moving"]].to_csv(flow_path, index=False)
    print(f"  Saved → {flow_path}")

    # ── Metric 3: Dead Zones ──
    print("\n── Dead Zone Detection ──────────────────────────────")
    dead = compute_dead_zones(df, args.min_dead)
    if dead.empty:
        print("  No dead zones found (all cells had activity)")
    else:
        print(dead.sort_values("duration_frames", ascending=False).head(10).to_string(index=False))
    dead_path = out_dir / "dead_zones.csv"
    dead.to_csv(dead_path, index=False)
    print(f"  Saved → {dead_path}")

    # ── Metric 4: Surge ──
    print("\n── Pedestrian Surge Predictor ───────────────────────")
    surge = compute_surge_predictor(df, args.surge_window)
    surge_events = surge[surge["surge"]]
    print(f"  Total surge frames detected: {len(surge_events):,}")
    if not surge_events.empty:
        top = surge_events.groupby("area")["surge"].sum().sort_values(ascending=False)
        print("  Surge frequency by zone:")
        print(top.head(10).to_string())
    surge_path = out_dir / "surge.csv"
    surge[["timestamp","area","passengers","roll_mean","roll_std","surge"]].to_csv(surge_path, index=False)
    print(f"  Saved → {surge_path}")

    # ── Master CSV ──
    master_path = out_dir / "advanced_metrics.csv"
    df.to_csv(master_path, index=False)
    print(f"\n  Master CSV → {master_path}")
    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    main()