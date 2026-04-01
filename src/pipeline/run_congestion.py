"""
congestion_analysis.py
----------------------
VisionX — Congestion Analysis for Dragon Lake dataset

Zone methodology:
  Zones are computed automatically from frame dimensions as a 4×4 grid.
  Zone function (parking bay / throughput corridor / transition) is derived
  from trajectory statistics — dwell time, speed, entry/exit frequency,
  direction variance — and clustered with k-means (k=3).

  This approach follows:
    Wang, X., Ma, X., & Grimson, W.E.L. (2009). Unsupervised Activity
    Perception in Crowded and Complicated Scenes Using Hierarchical Bayesian
    Models. IEEE TPAMI 31(3), 539–555.

    Morris, B.T. & Trivedi, M.M. (2011). Trajectory Learning for Activity
    Understanding: Unsupervised, Multilevel, and Long-Term Adaptive Approach.
    IEEE TPAMI 33(11), 2287–2301.

  Thresholds are calibrated per zone at the 75th percentile of observed
  vehicle counts, not a global constant. This means a corridor zone's
  threshold reflects corridor behaviour, and a parking bay's reflects
  bay behaviour — they have different baselines.

  No floor plan of Dragon Lake is required. The data labels itself.
"""

import json
import re
from collections import defaultdict
from pathlib import Path  # ← was missing

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# ZONE TYPE LABELS
# ─────────────────────────────────────────────────────────────────────────────

ZONE_TYPE_LABELS = {
    "parking_bay": "Parking bay — high dwell, low speed, low throughput",
    "corridor": "Throughput corridor — low dwell, high speed, high entry/exit",
    "transition": "Transition/manoeuvre area — mixed dwell, high direction variance",
}

# Severity levels derived from occupancy vs zone-specific threshold
# density_ratio = vehicle_count / zone_threshold
# < 1.0  → Clear
# >= 1.0 → Warning
# >= 1.5 → Critical
SEVERITY_THRESHOLDS = {"critical": 1.5, "warning": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CongestionAnalyzer:
    def __init__(
        self,
        base_dir,
        scene_name="DJI_0012",
        frame_w=3840,
        frame_h=2160,
        grid_rows=4,
        grid_cols=4,
    ):
        self.base_dir = Path(base_dir)
        self.scene_name = scene_name
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.input_file = (
            self.base_dir / "data" / "processed" / scene_name / "detections.json"
        )
        self.output_dir = (
            self.base_dir / "data" / "processed" / scene_name / "dashboard_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ZONES = self._compute_zones()
        self.VEHICLE_CLASSES = ["car", "van", "truck", "bus", "vehicle"]

    # ── Zone grid ─────────────────────────────────────────────────────────────

    def _compute_zones(self):
        """
        Divide the frame into a grid_rows × grid_cols grid.
        Each zone = [x_min, y_min, x_max, y_max] in pixel coordinates.
        Named Z_{row}_{col} so spatial position is encoded in the name.
        """
        cw = self.frame_w / self.grid_cols
        ch = self.frame_h / self.grid_rows
        zones = {}

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                zones[f"Z_{r}_{c}"] = [
                    c * cw,
                    r * ch,
                    (c + 1) * cw,
                    (r + 1) * ch,
                ]

        return zones

    def assign_zone(self, center_px):
        """Map a detection centroid to its zone name."""
        if center_px is None or len(center_px) < 2:
            return "Unknown"

        try:
            x, y = float(center_px[0]), float(center_px[1])
        except (TypeError, ValueError):
            return "Unknown"

        for zone_name, (x0, y0, x1, y1) in self.ZONES.items():
            if x0 <= x < x1 and y0 <= y < y1:
                return zone_name

        return "Outside"

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self):
        """
        Parse detections.json into a flat list of detection dicts.

        Robustly handles:
          1. List of frame dicts each with a 'detections' key
          2. Flat list of detection dicts
          3. Single frame dict
          4. Nested wrappers with arbitrary keys
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        detections = []

        def _extract(item, frame_id=None):
            if not isinstance(item, dict):
                return

            cls = str(
                item.get("class_name", item.get("class", item.get("label", "")))
            ).strip().lower()

            if cls not in self.VEHICLE_CLASSES:
                return

            center = item.get("center_px", item.get("center", item.get("centroid")))
            bbox = item.get("bbox", item.get("bounding_box", item.get("box")))

            # Fallback: derive center from bbox if missing
            if center is None and bbox is not None and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = map(float, bbox[:4])
                    center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                except (TypeError, ValueError):
                    center = None

            detections.append(
                {
                    "frame_id": frame_id
                    if frame_id is not None
                    else item.get("frame_idx", item.get("frame_id", 0)),
                    "track_id": item.get(
                        "track_id", item.get("id", item.get("object_id"))
                    ),
                    "class_name": cls,
                    "center_px": center,
                    "bbox": bbox,
                    "confidence": float(
                        item.get("confidence", item.get("conf", item.get("score", 0.0)))
                    ),
                }
            )

        def _looks_like_detection(obj):
            if not isinstance(obj, dict):
                return False
            has_class = any(k in obj for k in ("class_name", "class", "label"))
            has_geom = any(
                k in obj
                for k in ("center_px", "center", "centroid", "bbox", "bounding_box", "box")
            )
            return has_class and has_geom

        def _walk(obj, inherited_frame_id=None):
            if isinstance(obj, list):
                for item in obj:
                    _walk(item, inherited_frame_id)
                return

            if not isinstance(obj, dict):
                return

            current_frame_id = obj.get(
                "frame_idx",
                obj.get(
                    "frame_id",
                    inherited_frame_id if inherited_frame_id is not None else 0,
                ),
            )

            # Standard frame wrapper: {"frame_idx": ..., "vehicles": [...]}
            if "vehicles" in obj and isinstance(obj["vehicles"], list):
                for d in obj["vehicles"]:
                    _extract(d, current_frame_id)

            # Standard frame wrapper: {"frame_idx": ..., "detections": [...]}
            if "detections" in obj and isinstance(obj["detections"], list):
                for d in obj["detections"]:
                    _extract(d, current_frame_id)

            # Flat detection dict
            if _looks_like_detection(obj):
                _extract(obj, current_frame_id)

            # Recurse into all nested values to handle arbitrary wrapper keys
            for key, value in obj.items():
                if key in ("detections", "vehicles"):
                    continue
                if isinstance(value, (dict, list)):
                    _walk(value, current_frame_id)

        _walk(raw)

        # De-duplicate in case the recursive walk finds the same detection twice
        deduped = []
        seen = set()

        for d in detections:
            key = (
                d["frame_id"],
                d["track_id"],
                d["class_name"],
                tuple(d["center_px"]) if isinstance(d["center_px"], list) else d["center_px"],
            )
            if key not in seen:
                seen.add(key)
                deduped.append(d)

        return deduped

    # ── Trajectory feature extraction ─────────────────────────────────────────

    def extract_trajectory_features(self, df):
        """
        For each zone, compute 5 trajectory-level features that discriminate
        zone function without requiring layout knowledge:

          1. mean_dwell    — average frames a tracked object stays in the zone.
                            High = parking bay, low = through-traffic corridor.
          2. mean_speed    — average inter-frame centroid displacement (pixels).
                            High = corridor, low = parked vehicle.
          3. entry_exit_hz — unique track IDs entering zone per frame.
                            High = corridor or entry/exit point.
          4. dir_variance  — variance of movement direction (radians).
                            High = manoeuvre area, Low = directional lane.
          5. dwell_std     — std dev of dwell times.
                            High = mixed use (some park, some pass through).

        Returns a dict: zone_name → feature vector [f1..f5]
        """
        zone_tracks = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            zone = row["zone_id"]
            tid = row["track_id"]
            cp = row["center_px"]
            fid = row["frame_id"]

            if tid is None or cp is None or len(cp) < 2:
                continue

            try:
                cx, cy = float(cp[0]), float(cp[1])
            except (TypeError, ValueError):
                continue

            zone_tracks[zone][tid].append((int(fid), cx, cy))

        total_frames = df["frame_id"].nunique()
        features = {}

        for zone in self.ZONES:
            tracks = zone_tracks.get(zone, {})
            if not tracks:
                features[zone] = [0.0, 0.0, 0.0, 0.0, 0.0]
                continue

            dwell_times, speeds, directions = [], [], []

            for tid, obs in tracks.items():
                obs_sorted = sorted(obs, key=lambda x: x[0])
                dwell_times.append(len(obs_sorted))

                for i in range(1, len(obs_sorted)):
                    dx = obs_sorted[i][1] - obs_sorted[i - 1][1]
                    dy = obs_sorted[i][2] - obs_sorted[i - 1][2]
                    speed = np.sqrt(dx**2 + dy**2)
                    speeds.append(speed)
                    if speed > 0.5:  # ignore sub-pixel jitter
                        directions.append(np.arctan2(dy, dx))

            features[zone] = [
                float(np.mean(dwell_times)),                                    # 1 mean_dwell
                float(np.mean(speeds)) if speeds else 0.0,                     # 2 mean_speed
                len(tracks) / max(total_frames, 1),                            # 3 entry_exit_hz
                float(np.var(directions)) if len(directions) > 1 else 0.0,    # 4 dir_variance
                float(np.std(dwell_times)) if len(dwell_times) > 1 else 0.0,  # 5 dwell_std
            ]

        return features

    # ── Grid-position fallback ────────────────────────────────────────────────

    def _grid_based_zone_types(self):
        """
        Fallback when trajectory clustering is unavailable (sparse data, sklearn
        error, or fewer than 3 populated zones).

        Heuristic: corners → transition, edges → corridor, interior → parking_bay.
        This matches the typical aerial-view layout of a surface car park.
        """
        zone_types = {}
        zone_features = {}

        for zone_name in self.ZONES:
            m = re.match(r"Z_(\d+)_(\d+)", zone_name)
            if not m:
                ztype = "transition"
            else:
                r, c = int(m.group(1)), int(m.group(2))
                on_row_edge = r in (0, self.grid_rows - 1)
                on_col_edge = c in (0, self.grid_cols - 1)
                if on_row_edge and on_col_edge:
                    ztype = "transition"   # corner — turning / manoeuvre area
                elif on_row_edge or on_col_edge:
                    ztype = "corridor"     # edge lane — entry/exit traffic
                else:
                    ztype = "parking_bay"  # interior — parked vehicles

            zone_types[zone_name] = ztype
            zone_features[zone_name] = {
                "mean_dwell_frames": 0.0,
                "mean_speed_px": 0.0,
                "entry_exit_hz": 0.0,
                "direction_variance": 0.0,
                "dwell_std": 0.0,
                "zone_type": ztype,
                "zone_type_label": ZONE_TYPE_LABELS[ztype],
                "characterisation_method": "grid_position_fallback",
            }

        return zone_types, zone_features

    # ── Zone characterisation via k-means ────────────────────────────────────

    def characterise_zones(self, df):
        """
        Cluster zones into 3 functional types using k-means on trajectory
        features. Falls back to grid-position heuristics if clustering cannot
        run (sparse data, import error, fewer than 3 populated zones, etc.).
        """
        # Guard: need at least 3 zones with observations to form 3 clusters
        observed_zones = df["zone_id"].nunique() if not df.empty else 0
        if observed_zones < 3:
            print(
                f"  WARNING: Only {observed_zones} zone(s) have observations — "
                "skipping trajectory clustering, using grid-position fallback."
            )
            return self._grid_based_zone_types()

        try:
            feature_map = self.extract_trajectory_features(df)
            zone_names = list(self.ZONES.keys())
            feature_matrix = np.array([feature_map[z] for z in zone_names])

            # Normalise — critical because dwell (frames) and speed (pixels) are
            # on completely different scales.
            scaler = StandardScaler()
            X = scaler.fit_transform(feature_matrix)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Map cluster IDs to zone types by mean_dwell (feature index 0)
            cluster_dwell = {
                c: np.mean([feature_matrix[i][0] for i, l in enumerate(labels) if l == c])
                for c in range(3)
            }
            sorted_by_dwell = sorted(cluster_dwell, key=cluster_dwell.get)
            type_map = {
                sorted_by_dwell[0]: "corridor",     # lowest dwell
                sorted_by_dwell[1]: "transition",   # middle dwell
                sorted_by_dwell[2]: "parking_bay",  # highest dwell
            }

            zone_types = {z: type_map[labels[i]] for i, z in enumerate(zone_names)}
            zone_features = {
                z: {
                    "mean_dwell_frames": round(feature_map[z][0], 2),
                    "mean_speed_px": round(feature_map[z][1], 2),
                    "entry_exit_hz": round(feature_map[z][2], 4),
                    "direction_variance": round(feature_map[z][3], 4),
                    "dwell_std": round(feature_map[z][4], 2),
                    "zone_type": zone_types[z],
                    "zone_type_label": ZONE_TYPE_LABELS[zone_types[z]],
                    "characterisation_method": "trajectory_kmeans_k3",
                }
                for z in zone_names
            }
            return zone_types, zone_features

        except Exception as exc:
            print(
                f"  WARNING: Trajectory clustering failed ({exc}). "
                "Using grid-position fallback."
            )
            return self._grid_based_zone_types()

    # ── Per-zone p75 thresholds ──────────────────────────────────────────────

    def calibrate_thresholds(self, zone_frame_stats, zone_types):
        """
        Compute the 75th-percentile vehicle count per zone as its congestion
        threshold.

        Additionally compute per-cluster thresholds so that zones with no
        observed traffic still get a sensible fallback from their cluster peers.
        """
        thresholds = {}

        # Per-cluster p75 as fallback for zones with sparse data
        cluster_counts = defaultdict(list)
        for zone, ztype in zone_types.items():
            zone_data = zone_frame_stats[zone_frame_stats["zone_id"] == zone]["vehicle_count"]
            if len(zone_data) >= 10:
                cluster_counts[ztype].extend(zone_data.tolist())

        cluster_p75 = {
            ztype: float(np.percentile(counts, 75)) if counts else 1.0
            for ztype, counts in cluster_counts.items()
        }

        for zone in self.ZONES:
            zone_data = zone_frame_stats[
                zone_frame_stats["zone_id"] == zone
            ]["vehicle_count"]

            if len(zone_data) >= 10:
                t = float(np.percentile(zone_data, 75))
            else:
                ztype = zone_types.get(zone, "parking_bay")
                t = cluster_p75.get(ztype, 1.0)

            thresholds[zone] = max(t, 1.0)  # never let threshold drop to 0

        return thresholds, cluster_p75

    # ── Severity ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_severity(vehicle_count, threshold):
        """
        Severity is derived from density_ratio = vehicle_count / threshold.
        """
        if threshold <= 0:
            return "Clear", 0.0

        ratio = vehicle_count / threshold
        if ratio >= SEVERITY_THRESHOLDS["critical"]:
            return "Critical", round(ratio, 3)
        if ratio >= SEVERITY_THRESHOLDS["warning"]:
            return "Warning", round(ratio, 3)
        return "Clear", round(ratio, 3)

    # ── Main analysis ────────────────────────────────────────────────────────

    def analyze(self):
        print("Loading detection data...")
        detections = self.load_data()
        print(f"  Loaded {len(detections)} vehicle detections")

        df = pd.DataFrame(
            detections,
            columns=["frame_id", "track_id", "class_name", "center_px", "bbox", "confidence"],
        )

        if df.empty:
            raise ValueError(
                f"No usable vehicle detections were parsed from {self.input_file}. "
                "Check whether the file path is correct and whether detections are nested "
                "under a different top-level structure."
            )

        print("Assigning detections to zones...")
        df["zone_id"] = df["center_px"].apply(self.assign_zone)
        df = df[df["zone_id"].isin(self.ZONES.keys())].copy()

        if df.empty:
            raise ValueError(
                "No detections fell within defined zones. "
                "Check that center_px coordinates match the frame dimensions "
                f"({self.frame_w}×{self.frame_h}). "
                "If detections are in a downscaled coordinate space, pass the "
                "correct frame_w/frame_h to CongestionAnalyzer()."
            )

        print(f"  Zone distribution:\n{df['zone_id'].value_counts().to_string()}")

        # ── Step 1: Zone-frame statistics ────────────────────────────────────
        print("\nComputing zone-frame statistics...")
        zone_frame_stats = (
            df.groupby(["frame_id", "zone_id"])
            .agg(
                vehicle_count=("track_id", "count"),
                avg_confidence=("confidence", "mean"),
                class_dist=("class_name", lambda x: x.value_counts().to_dict()),
            )
            .reset_index()
        )

        # ── Step 2: Trajectory-based zone characterisation ──────────────────
        print("Characterising zones from trajectory data (Wang et al. 2009)...")
        zone_types, zone_features = self.characterise_zones(df)
        for zone, ztype in zone_types.items():
            print(f"  {zone} → {ztype}")

        # ── Step 3: Per-zone p75 threshold calibration ──────────────────────
        print("\nCalibrating per-zone thresholds (p75 per zone)...")
        thresholds, cluster_p75 = self.calibrate_thresholds(zone_frame_stats, zone_types)
        for zone, t in thresholds.items():
            print(f"  {zone} threshold = {t:.1f} ({zone_types[zone]})")

        # ── Step 4: Apply severity per row ──────────────────────────────────
        zone_frame_stats["threshold"] = zone_frame_stats["zone_id"].map(thresholds)
        zone_frame_stats["density_ratio"] = (
            zone_frame_stats["vehicle_count"] / zone_frame_stats["threshold"]
        ).round(3)
        zone_frame_stats[["severity", "density_ratio"]] = zone_frame_stats.apply(
            lambda row: pd.Series(
                self.compute_severity(row["vehicle_count"], row["threshold"])
            ),
            axis=1,
        )
        zone_frame_stats["is_congested"] = zone_frame_stats["density_ratio"] >= 1.0

        # ── Step 5: Zone summary ─────────────────────────────────────────────
        total_frames = zone_frame_stats["frame_id"].nunique()

        zone_summary_rows = []
        for zone in self.ZONES:
            zdf = zone_frame_stats[zone_frame_stats["zone_id"] == zone]
            if zdf.empty:
                continue

            cf = int(zdf["is_congested"].sum())
            zone_summary_rows.append(
                {
                    "zone_id": zone,
                    "zone_type": zone_types.get(zone, "unknown"),
                    "zone_type_label": ZONE_TYPE_LABELS.get(zone_types.get(zone, ""), ""),
                    "threshold": round(thresholds[zone], 1),
                    "avg_vehicle_count": round(float(zdf["vehicle_count"].mean()), 1),
                    "max_vehicle_count": int(zdf["vehicle_count"].max()),
                    "congested_frames": cf,
                    "total_frames": total_frames,
                    "congestion_rate_pct": round(cf / total_frames * 100, 2),
                    "avg_density_ratio": round(float(zdf["density_ratio"].mean()), 3),
                    "max_density_ratio": round(float(zdf["density_ratio"].max()), 3),
                    "is_chronic_hotspot": (cf / total_frames) >= 0.40,
                }
            )

        zone_summary = pd.DataFrame(zone_summary_rows).sort_values(
            "congestion_rate_pct", ascending=False
        )

        # ── Step 6: Timeline data ────────────────────────────────────────────
        timeline_data = (
            zone_frame_stats.groupby("frame_id")
            .agg(
                total_vehicles=("vehicle_count", "sum"),
                congested_zones=("is_congested", "sum"),
                critical_zones=("severity", lambda x: (x == "Critical").sum()),
            )
            .reset_index()
        )

        # ── Step 7: KPI snapshot from latest frame ──────────────────────────
        latest_frame_id = zone_frame_stats["frame_id"].max()
        latest = zone_frame_stats[zone_frame_stats["frame_id"] == latest_frame_id]
        peak_frame_id = timeline_data.loc[
            timeline_data["total_vehicles"].idxmax(), "frame_id"
        ]

        kpi_metrics = {
            "total_vehicles_now": int(latest["vehicle_count"].sum()),
            "congested_zones_now": int(latest["is_congested"].sum()),
            "critical_zones_now": int((latest["severity"] == "Critical").sum()),
            "total_zones": len(self.ZONES),
            "peak_vehicles": int(timeline_data["total_vehicles"].max()),
            "peak_frame_id": int(peak_frame_id),
            "chronic_hotspot_count": int(zone_summary["is_chronic_hotspot"].sum()),
            "total_frames_analyzed": total_frames,
            "threshold_method": "p75_per_zone",
            "zone_characterisation": "trajectory_kmeans_k3",
        }

        # ── Step 8: Alarms — only Critical zones ─────────────────────────────
        alarms = []
        for _, row in latest.iterrows():
            if row["severity"] == "Critical":
                alarms.append(
                    {
                        "zone_id": row["zone_id"],
                        "zone_type": zone_types.get(row["zone_id"], "unknown"),
                        "vehicle_count": int(row["vehicle_count"]),
                        "threshold": round(float(row["threshold"]), 1),
                        "density_ratio": round(float(row["density_ratio"]), 3),
                        "severity": "Critical",
                        "frame_id": int(latest_frame_id),
                    }
                )

        # ── Step 9: Save outputs ─────────────────────────────────────────────
        print("\nSaving outputs...")

        outputs = {
            "zone_frame_stats.json": zone_frame_stats.to_dict(orient="records"),
            "zone_summary.json": zone_summary.to_dict(orient="records"),
            "timeline_data.json": timeline_data.to_dict(orient="records"),
            "kpi_metrics.json": kpi_metrics,
            "alarms.json": alarms,
            "zone_thresholds.json": thresholds,
            "zone_metadata.json": zone_features,
            "zone_config.json": {
                "zones": self.ZONES,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "frame_w": self.frame_w,
                "frame_h": self.frame_h,
                "vehicle_classes": self.VEHICLE_CLASSES,
                "threshold_method": "p75_per_zone",
                "cluster_p75": cluster_p75,
                "methodology": (
                    "Trajectory-based zone characterisation. "
                    "Wang et al. TPAMI 2009 / Morris & Trivedi TPAMI 2011. "
                    "k-means k=3 on [dwell_time, speed, entry_exit_hz, "
                    "direction_variance, dwell_std]. "
                    "Thresholds: 75th percentile per zone, "
                    "cluster-peer fallback for sparse zones."
                ),
            },
        }

        for filename, data in outputs.items():
            path = self.output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  Saved {filename}")

        # ── Print summary ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\n  Detections loaded:    {len(df)}")
        print(f"  Frames analyzed:      {total_frames}")
        print(f"  Zones:                {self.grid_rows}x{self.grid_cols} = {len(self.ZONES)}")
        print("\n  Zone types discovered:")
        for ztype in ["parking_bay", "corridor", "transition"]:
            count = sum(1 for v in zone_types.values() if v == ztype)
            print(f"    {ztype:15s}: {count} zones")

        print("\n  KPI snapshot (latest frame):")
        print(f"    Vehicles now:         {kpi_metrics['total_vehicles_now']}")
        print(f"    Congested zones:      {kpi_metrics['congested_zones_now']}/{len(self.ZONES)}")
        print(f"    Critical zones:       {kpi_metrics['critical_zones_now']}")
        print(f"    Chronic hotspots:     {kpi_metrics['chronic_hotspot_count']}")

        print(f"\n  Active critical alarms: {len(alarms)}")
        for a in alarms:
            print(
                f"    {a['zone_id']} ({a['zone_type']}): "
                f"{a['vehicle_count']} veh, density {a['density_ratio']:.2f}x"
            )

        print(f"\n  Output: {self.output_dir}")
        print("=" * 60)

        return {
            "zone_frame_stats": zone_frame_stats,
            "zone_summary": zone_summary,
            "timeline_data": timeline_data,
            "kpi_metrics": kpi_metrics,
            "alarms": alarms,
            "zone_types": zone_types,
            "zone_features": zone_features,
            "thresholds": thresholds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = r"D:\VisionX\SpringHackathon-2026-VisionX"

    analyzer = CongestionAnalyzer(
        base_dir=base_dir,
        scene_name="DJI_0012",
        frame_w=3840,
        frame_h=2160,
        grid_rows=4,
        grid_cols=4,
    )
    results = analyzer.analyze()