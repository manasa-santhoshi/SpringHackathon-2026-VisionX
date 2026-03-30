"""
Parking Lot Analytics Dashboard.

Four tabs:
1. Live Demo — real-time YOLO inference on DLP video with live metrics
2. Vehicle Analytics — metrics from the batch detection/tracking pipeline (DLP)
3. Pedestrian Analytics — person counting and Parking Stress Index (PSI)
4. Anomaly Detection — multi-camera skeleton-based anomaly detection (CHAD)

Usage:
    streamlit run dashboard/app.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "anomaly"


@st.cache_data
def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_available_scenes() -> list[str]:
    """Find scenes with processed data."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted(
        d.name for d in PROCESSED_DIR.iterdir()
        if d.is_dir() and (d / "vehicle_count.json").exists()
    )


def get_available_anomaly_models() -> list[str]:
    """Find trained anomaly detection models."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "evaluation.json").exists()
    )


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


# --- Page config ---
st.set_page_config(page_title="Parking Analytics", page_icon="P", layout="wide")
st.title("Parking Lot Analytics Dashboard")

# --- Tabs ---
tab_live, tab_vehicle, tab_pedestrian, tab_anomaly_live, tab_anomaly = st.tabs(
    ["Live Demo", "Vehicle Analytics", "Pedestrian Analytics",
     "Anomaly Demo", "Anomaly Detection"]
)

# ============================================================
# TAB 1: Live Demo (real-time inference)
# ============================================================
with tab_live:
    st.subheader("Real-Time Vehicle Detection & Tracking")

    # --- Config sidebar (inside tab to avoid polluting other tabs) ---
    DLP_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "DLP" / "raw"
    DLP_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "DLP" / "json"

    # Find available videos
    available_videos = sorted(DLP_RAW_DIR.glob("*.MOV")) if DLP_RAW_DIR.exists() else []

    if not available_videos:
        st.info(
            "No DLP video files found. Place `.MOV` files in `data/raw/DLP/raw/`.\n\n"
            "Download from the [DLP dataset site](https://sites.google.com/berkeley.edu/dlp-dataset)."
        )
    else:
        # Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            video_file = st.selectbox(
                "Video",
                available_videos,
                format_func=lambda p: p.name,
                key="live_video",
            )
        with ctrl_col2:
            frame_skip = st.slider("Process every Nth frame", 1, 10, 3, key="live_skip")
        with ctrl_col3:
            model_path = st.text_input(
                "Model",
                value=str(PROJECT_ROOT / "models" / "yolo11n-visdrone" / "weights" / "bestVisDrone.pt"),
                key="live_model",
            )

        # Session state
        if "live_running" not in st.session_state:
            st.session_state.live_running = False

        start_col, stop_col, status_col = st.columns([1, 1, 2])
        with start_col:
            if st.button("Start", key="live_start", use_container_width=True):
                st.session_state.live_running = True
        with stop_col:
            if st.button("Stop", key="live_stop", use_container_width=True):
                st.session_state.live_running = False

        st.markdown("---")

        # Layout: video on left, metrics on right
        video_col, metrics_col = st.columns([3, 2])

        with video_col:
            frame_placeholder = st.empty()
            fps_placeholder = st.empty()
            progress_placeholder = st.empty()

        with metrics_col:
            kpi_placeholder = st.empty()
            occ_chart_placeholder = st.empty()
            count_chart_placeholder = st.empty()
            dwell_placeholder = st.empty()

        # --- Run inference ---
        if st.session_state.live_running and video_file:
            # Lazy imports to avoid slowing tab load
            sys.path.insert(0, str(PROJECT_ROOT / "dlp-dataset"))
            from ultralytics import YOLO
            from src.detection.yolo_detector import _detect_class_map
            from src.pipeline.realtime import (
                MetricsAccumulator,
                draw_detections,
                extract_detections_from_result,
            )
            from src.pipeline.homography import load_homography, compute_homography
            from src.pipeline.run import get_parking_spaces

            # Load homography
            scene_name = video_file.stem
            homography_cache = PROCESSED_DIR / scene_name / "homography.npy"
            xml_path = DLP_RAW_DIR / f"{scene_name}_data.xml"

            H = None
            if homography_cache.exists():
                H = load_homography(str(homography_cache))
            elif xml_path.exists():
                H = compute_homography(str(xml_path))

            if H is None:
                st.error("Cannot load homography. Ensure XML annotation exists.")
                st.session_state.live_running = False
            else:
                # Load parking spaces and model
                try:
                    parking_spaces = get_parking_spaces()
                except Exception as e:
                    st.error(f"Cannot load parking spaces: {e}")
                    st.session_state.live_running = False
                    parking_spaces = None

                if parking_spaces is not None:
                    model = YOLO(model_path)
                    vehicle_classes = _detect_class_map(model)
                    accumulator = MetricsAccumulator(H, parking_spaces)

                    cap = cv2.VideoCapture(str(video_file))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                    frame_idx = 0
                    processed = 0

                    while cap.isOpened() and st.session_state.live_running:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break

                        frame_idx += 1

                        # Skip frames for performance
                        if frame_idx % frame_skip != 0:
                            continue

                        t0 = time.time()
                        timestamp = frame_idx / fps

                        # Run YOLO tracking on single frame
                        results = model.track(
                            source=frame_bgr,
                            persist=True,
                            imgsz=1920,
                            conf=0.25,
                            classes=list(vehicle_classes.keys()),
                            verbose=False,
                        )

                        # Extract detections
                        det = extract_detections_from_result(
                            results[0], frame_idx, timestamp, vehicle_classes
                        )

                        # Update metrics
                        accumulator.add_frame(det)
                        snapshot = accumulator.get_snapshot()

                        # Draw and display frame
                        annotated = draw_detections(frame_bgr, det, scale=0.5)
                        frame_placeholder.image(annotated, channels="RGB")

                        # FPS display
                        proc_time = time.time() - t0
                        current_fps = 1.0 / proc_time if proc_time > 0 else 0
                        fps_placeholder.caption(
                            f"Frame {frame_idx}/{total_frames} | "
                            f"{current_fps:.1f} FPS | "
                            f"Time: {timestamp:.1f}s"
                        )
                        progress_placeholder.progress(
                            min(frame_idx / total_frames, 1.0)
                        )

                        # Update KPI cards
                        vc = snapshot["vehicle_count"]
                        occ = snapshot["occupancy"]
                        dw = snapshot["dwell"]
                        ee = snapshot["entry_exit"]

                        with kpi_placeholder.container():
                            k1, k2 = st.columns(2)
                            with k1:
                                st.metric("Unique Vehicles", vc["total_unique"])
                                st.metric("In Frame", vc["current_frame_count"])
                            with k2:
                                occ_pct = round(
                                    occ["current_occupied"] / occ["total_spaces"] * 100, 1
                                ) if occ["total_spaces"] else 0
                                st.metric(
                                    "Occupancy",
                                    f"{occ['current_occupied']}/{occ['total_spaces']}",
                                    f"{occ_pct}%",
                                )
                                st.metric(
                                    "Entries / Exits",
                                    f"{ee['entry_count']} / {ee['exit_count']}",
                                )

                        # Update occupancy chart (every 5 processed frames)
                        processed += 1
                        if processed % 5 == 0 and occ["timestamps"]:
                            fig_occ = go.Figure()
                            fig_occ.add_trace(go.Scatter(
                                x=occ["timestamps"],
                                y=occ["occupied"],
                                mode="lines",
                                fill="tozeroy",
                                fillcolor="rgba(255, 99, 71, 0.3)",
                                line=dict(color="tomato"),
                                name="Occupied",
                            ))
                            fig_occ.update_layout(
                                title="Occupancy Over Time",
                                xaxis_title="Time (s)",
                                yaxis_title="Occupied Spots",
                                height=250,
                                margin=dict(t=30, b=30),
                                showlegend=False,
                            )
                            occ_chart_placeholder.plotly_chart(
                                fig_occ, use_container_width=True, key=f"occ_chart_{frame_idx}"
                            )

                        # Dwell info
                        if dw["stats"]:
                            dwell_placeholder.caption(
                                f"Completed dwells: {dw['completed']} | "
                                f"Active parked: {dw['active_parked']} | "
                                f"Avg: {format_duration(dw['stats']['mean_sec'])}"
                            )

                    cap.release()
                    st.session_state.live_running = False
                    st.success("Video processing complete.")


# ============================================================
# TAB 2: Vehicle Analytics (existing functionality)
# ============================================================
with tab_vehicle:
    scenes = get_available_scenes()
    if not scenes:
        st.info(
            "No processed data found. Run the pipeline first:\n\n"
            "```\npython -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV\n```"
        )
    else:
        scene = st.selectbox("Scene", scenes, key="vehicle_scene")
        scene_dir = PROCESSED_DIR / scene

        # --- Load data ---
        vehicle_count = load_json(scene_dir / "vehicle_count.json")
        occupancy = load_json(scene_dir / "occupancy_timeline.json")
        dwell = load_json(scene_dir / "dwell_times.json")
        entry_exit = load_json(scene_dir / "entry_exit.json")

        # --- Row 1: KPI Cards ---
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if vehicle_count:
                st.metric("Total Vehicles Detected", vehicle_count["total_unique"])
                by_class = vehicle_count.get("by_class", {})
                if by_class:
                    breakdown = ", ".join(f"{v} {k}s" for k, v in sorted(by_class.items()))
                    st.caption(breakdown)

        with col2:
            if occupancy:
                avg_occupied = round(np.mean(occupancy["occupied"]), 1)
                total = occupancy["total_spaces"]
                occupancy_pct = round(avg_occupied / total * 100, 1) if total else 0
                st.metric("Avg Occupancy", f"{avg_occupied}/{total}", f"{occupancy_pct}%")

        with col3:
            if dwell and dwell.get("stats"):
                mean_dwell = dwell["stats"]["mean_sec"]
                st.metric("Avg Dwell Time", format_duration(mean_dwell))
                st.caption(
                    f"Median: {format_duration(dwell['stats']['median_sec'])}, "
                    f"{dwell['stats']['count']} events"
                )

        with col4:
            if entry_exit:
                st.metric("Entries / Exits",
                           f"{entry_exit['entry_count']} / {entry_exit['exit_count']}")

        # --- Row 2: Occupancy Over Time ---
        if occupancy:
            st.markdown("---")
            st.subheader("Occupancy Over Time")

            show_areas = st.checkbox("Show by area", value=False)

            if show_areas and occupancy.get("by_area"):
                fig = go.Figure()
                for area_name, area_data in sorted(occupancy["by_area"].items()):
                    fig.add_trace(go.Scatter(
                        x=occupancy["timestamps"],
                        y=area_data["occupied"],
                        mode="lines",
                        name=f"Area {area_name} ({area_data['total']} spots)",
                    ))
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Occupied Spots",
                    height=400,
                )
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=occupancy["timestamps"],
                    y=occupancy["occupied"],
                    mode="lines",
                    name="Occupied",
                    fill="tozeroy",
                    fillcolor="rgba(255, 99, 71, 0.3)",
                    line=dict(color="tomato"),
                ))
                fig.add_trace(go.Scatter(
                    x=occupancy["timestamps"],
                    y=occupancy["free"],
                    mode="lines",
                    name="Free",
                    fill="tozeroy",
                    fillcolor="rgba(60, 179, 113, 0.3)",
                    line=dict(color="mediumseagreen"),
                ))
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Parking Spots",
                    height=400,
                )

            st.plotly_chart(fig, use_container_width=True)

        # --- Row 3: Dwell Times + Entry/Exit ---
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Dwell Time Distribution")
            if dwell and dwell.get("dwell_times"):
                durations = [d["duration_sec"] for d in dwell["dwell_times"]]
                fig_dwell = px.histogram(
                    x=durations,
                    nbins=20,
                    labels={"x": "Duration (s)", "y": "Count"},
                    color_discrete_sequence=["steelblue"],
                )
                fig_dwell.update_layout(
                    xaxis_title="Duration (s)",
                    yaxis_title="Count",
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig_dwell, use_container_width=True)

                # Show by area
                if any(d.get("area") for d in dwell["dwell_times"]):
                    area_stats = {}
                    for d in dwell["dwell_times"]:
                        area = d.get("area", "Unknown")
                        if area not in area_stats:
                            area_stats[area] = []
                        area_stats[area].append(d["duration_sec"])
                    rows = []
                    for area, durs in sorted(area_stats.items()):
                        rows.append({
                            "Area": area,
                            "Count": len(durs),
                            "Mean": format_duration(np.mean(durs)),
                            "Median": format_duration(np.median(durs)),
                        })
                    st.dataframe(rows, use_container_width=True)
            else:
                st.info("No dwell time data available")

        with col_right:
            st.subheader("Cumulative Entries & Exits")
            if entry_exit and entry_exit.get("timeline"):
                timeline = entry_exit["timeline"]
                fig_ee = go.Figure()
                fig_ee.add_trace(go.Scatter(
                    x=timeline["timestamps"],
                    y=timeline["cumulative_entries"],
                    mode="lines",
                    name="Entries",
                    line=dict(color="dodgerblue"),
                ))
                fig_ee.add_trace(go.Scatter(
                    x=timeline["timestamps"],
                    y=timeline["cumulative_exits"],
                    mode="lines",
                    name="Exits",
                    line=dict(color="coral"),
                ))
                fig_ee.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Cumulative Count",
                    height=350,
                )
                st.plotly_chart(fig_ee, use_container_width=True)
            else:
                st.info("No entry/exit data available")

        # --- Row 4: Per-frame detection count ---
        if vehicle_count and vehicle_count.get("per_frame_counts"):
            st.markdown("---")
            st.subheader("Vehicles Detected Per Frame")

            frame_data = vehicle_count["per_frame_counts"]
            # Downsample for performance
            step = max(1, len(frame_data) // 500)
            sampled = frame_data[::step]

            fig_count = px.line(
                x=[f["timestamp"] for f in sampled],
                y=[f["count"] for f in sampled],
                labels={"x": "Time (s)", "y": "Vehicle Count"},
            )
            fig_count.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_count, use_container_width=True)

        # --- Evaluation results ---
        evaluation = load_json(scene_dir / "evaluation.json")
        if evaluation:
            st.markdown("---")
            st.subheader("Model Evaluation vs Ground Truth")

            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                uc = evaluation.get("unique_count", {})
                if uc:
                    st.markdown("**Unique Vehicle Count**")
                    st.markdown(f"- Detected: **{uc['detected_unique']}**")
                    st.markdown(f"- Ground Truth: **{uc['gt_total']}** "
                                f"({uc['gt_moving_vehicles']} moving + {uc['gt_static_obstacles']} static)")

            with eval_col2:
                fc = evaluation.get("frame_count", {}).get("stats", {})
                if fc:
                    st.markdown("**Per-Frame Count Accuracy**")
                    st.markdown(f"- MAE: **{fc['mae']}** vehicles")
                    st.markdown(f"- Median AE: **{fc['median_ae']}**")
                    st.markdown(f"- Max AE: **{fc['max_ae']}**")


# ============================================================
# TAB 3: Pedestrian Analytics (person counting + PSI)
# ============================================================
with tab_pedestrian:
    ped_scenes = get_available_scenes()
    if not ped_scenes:
        st.info(
            "No processed data found. Run the pipeline first:\n\n"
            "```\npython -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV\n```"
        )
    else:
        ped_scene = st.selectbox("Scene", ped_scenes, key="ped_scene")
        ped_scene_dir = PROCESSED_DIR / ped_scene

        person_count = load_json(ped_scene_dir / "person_count.json")
        psi_data = load_json(ped_scene_dir / "psi.json")

        if not person_count and not psi_data:
            st.info(
                "No pedestrian metrics found for this scene. "
                "Re-run the pipeline to generate them:\n\n"
                "```\npython -m src.pipeline.run --video data/raw/DLP/raw/"
                f"{ped_scene}.MOV\n```"
            )
        else:
            # --- KPI Cards ---
            st.markdown("---")
            kp1, kp2, kp3, kp4 = st.columns(4)

            with kp1:
                if person_count:
                    st.metric("Total Unique Persons", person_count["total_unique"])

            with kp2:
                if person_count and person_count.get("per_frame_counts"):
                    counts = [f["count"] for f in person_count["per_frame_counts"]]
                    st.metric("Avg Persons / Frame", f"{np.mean(counts):.1f}")

            with kp3:
                if person_count and person_count.get("per_frame_counts"):
                    counts = [f["count"] for f in person_count["per_frame_counts"]]
                    st.metric("Peak Simultaneous", int(max(counts)))

            with kp4:
                if psi_data and psi_data.get("zones"):
                    peak_zone = max(psi_data["zones"], key=lambda z: z["peak_psi"])
                    st.metric(
                        "Peak PSI",
                        f"{peak_zone['peak_psi']:.1f}",
                        f"{peak_zone['area']}",
                    )

            # --- Persons Over Time ---
            if person_count and person_count.get("per_frame_counts"):
                st.markdown("---")
                st.subheader("Persons Detected Over Time")

                frame_data = person_count["per_frame_counts"]
                step = max(1, len(frame_data) // 500)
                sampled = frame_data[::step]

                fig_persons = px.line(
                    x=[f["timestamp"] for f in sampled],
                    y=[f["count"] for f in sampled],
                    labels={"x": "Time (s)", "y": "Person Count"},
                )
                fig_persons.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_persons, use_container_width=True)

            # --- PSI Heatmap + Zone Table ---
            if psi_data and psi_data.get("zones"):
                st.markdown("---")
                st.subheader("Parking Stress Index (PSI)")
                st.caption(
                    "PSI combines pedestrian density, vehicle density, and their ratio "
                    "into a single 0-10 score per zone. Higher = more stressed."
                )

                psi_left, psi_right = st.columns([3, 2])

                zones = psi_data["zones"]
                grid_info = psi_data["grid"]
                n_rows = grid_info["rows"]
                n_cols = grid_info["cols"]

                with psi_left:
                    # Build heatmap grid
                    heatmap = np.zeros((n_rows, n_cols))
                    hover_text = [[""] * n_cols for _ in range(n_rows)]

                    for z in zones:
                        parts = z["area"].split("_")
                        r, c = int(parts[1]), int(parts[2])
                        heatmap[r, c] = z["avg_psi"]
                        hover_text[r][c] = (
                            f"{z['area']}<br>"
                            f"Avg PSI: {z['avg_psi']}<br>"
                            f"Peak PSI: {z['peak_psi']}<br>"
                            f"Avg peds: {z['avg_peds']}<br>"
                            f"Avg vehicles: {z['avg_vehicles']}"
                        )

                    fig_heatmap = go.Figure(go.Heatmap(
                        z=heatmap,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale="YlOrRd",
                        colorbar=dict(title="Avg PSI"),
                    ))
                    fig_heatmap.update_layout(
                        xaxis=dict(
                            title="Column",
                            tickvals=list(range(n_cols)),
                        ),
                        yaxis=dict(
                            title="Row",
                            tickvals=list(range(n_rows)),
                            autorange="reversed",
                        ),
                        height=380,
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                with psi_right:
                    # Zone details table sorted by avg PSI descending
                    table_rows = sorted(zones, key=lambda z: z["avg_psi"], reverse=True)
                    st.dataframe(
                        [
                            {
                                "Zone": z["area"],
                                "Avg PSI": z["avg_psi"],
                                "Peak PSI": z["peak_psi"],
                                "Avg Peds": z["avg_peds"],
                                "Avg Vehicles": z["avg_vehicles"],
                            }
                            for z in table_rows
                        ],
                        use_container_width=True,
                        height=380,
                    )


# ============================================================
# TAB 4: Anomaly Demo (real-time CHAD skeleton inference)
# ============================================================
with tab_anomaly_live:
    st.subheader("Real-Time Behavior Anomaly Detection")
    st.caption(
        "Plays a CHAD surveillance video while running skeleton-based anomaly detection "
        "frame-by-frame. Alerts are triggered when the anomaly score crosses the threshold."
    )

    # Check for trained model
    anomaly_model_dirs = []
    if MODELS_DIR.exists():
        anomaly_model_dirs = sorted(
            d.name for d in MODELS_DIR.iterdir()
            if d.is_dir() and (d / "best_model.pt").exists()
        )

    CHAD_META_DIR = PROJECT_ROOT / "data" / "raw" / "CHAD" / "CHAD_Meta"
    CHAD_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "CHAD" / "CHAD_Videos"

    if not anomaly_model_dirs:
        st.info(
            "No trained anomaly model found. Train first:\n\n"
            "```\npython -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta\n```"
        )
    elif not CHAD_META_DIR.exists():
        st.info("CHAD dataset not found. Place it in `data/raw/CHAD/CHAD_Meta/`.")
    else:
        # Controls
        anom_ctrl1, anom_ctrl2, anom_ctrl3 = st.columns([2, 2, 1])

        with anom_ctrl1:
            anom_model_name = st.selectbox(
                "Model", anomaly_model_dirs, key="anom_live_model"
            )

        # Find available videos (with both .pkl and .npy)
        ann_dir = CHAD_META_DIR / "annotations"
        label_dir = CHAD_META_DIR / "anomaly_labels"
        available_chad = sorted(
            p.stem for p in ann_dir.glob("*.pkl")
            if (label_dir / f"{p.stem}.npy").exists()
        )

        # Filter to those with video files
        available_with_video = [
            v for v in available_chad
            if (CHAD_VIDEO_DIR / f"{v}.mp4").exists()
        ]

        if not available_with_video:
            st.warning(
                "No CHAD videos with matching annotations found. "
                "Ensure `.mp4` files are in `data/raw/CHAD/CHAD_Videos/`."
            )
        else:
            with anom_ctrl2:
                chad_video_stem = st.selectbox(
                    "Video",
                    available_with_video,
                    key="anom_live_video",
                )

            with anom_ctrl3:
                anom_frame_skip = st.slider(
                    "Frame skip", 1, 5, 1, key="anom_live_skip"
                )

            # Session state
            if "anom_running" not in st.session_state:
                st.session_state.anom_running = False

            start_c, stop_c, _ = st.columns([1, 1, 3])
            with start_c:
                if st.button("Start", key="anom_start", use_container_width=True):
                    st.session_state.anom_running = True
            with stop_c:
                if st.button("Stop", key="anom_stop", use_container_width=True):
                    st.session_state.anom_running = False

            st.markdown("---")

            # Layout: video left, metrics right
            vid_col, score_col = st.columns([3, 2])

            with vid_col:
                anom_frame_ph = st.empty()
                anom_progress_ph = st.empty()

            with score_col:
                anom_kpi_ph = st.empty()
                anom_chart_ph = st.empty()
                anom_alert_ph = st.empty()

            # --- Run inference ---
            if st.session_state.anom_running and chad_video_stem:
                import cv2
                import time
                from src.anomaly.realtime import RealtimeAnomalyDetector

                model_path = str(MODELS_DIR / anom_model_name)
                detector = RealtimeAnomalyDetector(
                    model_dir=model_path,
                    data_root=str(CHAD_META_DIR),
                )
                detector.load_video(chad_video_stem)

                video_path = CHAD_VIDEO_DIR / f"{chad_video_stem}.mp4"
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                frame_idx = 0
                alert_count = 0
                processed = 0

                while cap.isOpened() and st.session_state.anom_running:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    if frame_idx % anom_frame_skip != 0:
                        frame_idx += 1
                        continue

                    t0 = time.time()

                    # Run anomaly detection on this frame
                    result = detector.process_frame(frame_idx)

                    # Draw frame with anomaly overlay
                    display = frame_bgr.copy()
                    h_frame, w_frame = display.shape[:2]

                    # Color border based on anomaly
                    if result.is_anomaly:
                        alert_count += 1
                        # Red border
                        cv2.rectangle(display, (0, 0), (w_frame - 1, h_frame - 1), (0, 0, 255), 6)
                        cv2.putText(
                            display, "ANOMALY DETECTED",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        )
                    elif result.gt_label == 1:
                        # Orange border for GT anomaly but model didn't flag
                        cv2.rectangle(display, (0, 0), (w_frame - 1, h_frame - 1), (0, 165, 255), 4)

                    # Score overlay
                    score_text = f"Score: {result.max_score:.4f} | Thr: {detector.threshold:.4f}"
                    cv2.putText(
                        display, score_text,
                        (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )

                    # Person count
                    cv2.putText(
                        display, f"Persons: {result.num_persons}",
                        (10, h_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    )

                    # Resize and display
                    scale = 0.6
                    display = cv2.resize(display, (int(w_frame * scale), int(h_frame * scale)))
                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    anom_frame_ph.image(display_rgb)

                    anom_progress_ph.progress(
                        min(frame_idx / max(total_frames, 1), 1.0),
                        text=f"Frame {frame_idx}/{total_frames} | "
                             f"{1.0 / max(time.time() - t0, 0.001):.1f} FPS",
                    )

                    # Update KPIs
                    with anom_kpi_ph.container():
                        ak1, ak2 = st.columns(2)
                        with ak1:
                            st.metric("Current Score", f"{result.max_score:.4f}")
                            st.metric("Persons in Frame", result.num_persons)
                        with ak2:
                            st.metric("Alerts Triggered", alert_count)
                            gt_text = "Anomalous" if result.gt_label == 1 else "Normal"
                            st.metric("Ground Truth", gt_text)

                    # Update score timeline chart (every 5 processed frames)
                    processed += 1
                    if processed % 5 == 0 and detector.score_timeline:
                        tl = detector.score_timeline
                        step_tl = max(1, len(tl) // 300)
                        sampled_scores = tl[::step_tl]
                        sampled_gt = detector.gt_timeline[::step_tl]

                        fig_score = go.Figure()
                        x_vals = list(range(0, len(tl), step_tl))

                        fig_score.add_trace(go.Scatter(
                            x=x_vals, y=sampled_scores,
                            mode="lines", name="Anomaly Score",
                            line=dict(color="steelblue", width=1.5),
                        ))

                        # Threshold line
                        fig_score.add_hline(
                            y=detector.threshold,
                            line_dash="dash", line_color="red",
                            annotation_text="Threshold",
                            annotation_position="top right",
                        )

                        # Highlight GT anomaly regions
                        gt_x = [x for x, g in zip(x_vals, sampled_gt) if g == 1]
                        gt_y = [s for s, g in zip(sampled_scores, sampled_gt) if g == 1]
                        if gt_x:
                            fig_score.add_trace(go.Scatter(
                                x=gt_x, y=gt_y,
                                mode="markers", name="GT Anomaly",
                                marker=dict(color="red", size=3, opacity=0.5),
                            ))

                        fig_score.update_layout(
                            xaxis_title="Frame",
                            yaxis_title="Score",
                            height=250,
                            margin=dict(t=10, b=30),
                            legend=dict(font=dict(size=9)),
                        )
                        anom_chart_ph.plotly_chart(
                            fig_score, use_container_width=True,
                            key=f"anom_chart_{frame_idx}",
                        )

                    # Alert panel
                    if result.is_anomaly:
                        anom_alert_ph.error(
                            f"ALERT: Anomaly detected at frame {frame_idx} "
                            f"(score {result.max_score:.4f} > threshold {detector.threshold:.4f})"
                        )

                    frame_idx += 1

                cap.release()
                st.session_state.anom_running = False
                st.success(
                    f"Video complete. {alert_count} anomaly alerts triggered "
                    f"across {frame_idx} frames."
                )


# ============================================================
# TAB 5: Anomaly Detection (CHAD multi-camera)
# ============================================================
with tab_anomaly:
    models = get_available_anomaly_models()
    if not models:
        st.info(
            "No anomaly detection results found. Train and evaluate the model first:\n\n"
            "```bash\n"
            "# Train\n"
            "python -m src.anomaly.train --data-root data/raw/CHAD/CHAD_Meta\n\n"
            "# Evaluate\n"
            "python -m src.anomaly.evaluate --model-dir models/anomaly/cam_1_2_3_4\n"
            "```"
        )
    else:
        model_name = st.selectbox("Model", models, key="anomaly_model")
        model_dir = MODELS_DIR / model_name

        eval_data = load_json(model_dir / "evaluation.json")
        config = load_json(model_dir / "config.json")

        if eval_data is None:
            st.error("Evaluation data not found.")
        else:
            overall = eval_data["overall"]
            threshold = eval_data.get("threshold", 0)

            # --- KPI Cards ---
            st.markdown("---")
            st.subheader("Model Performance")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric("AUC-ROC", f"{overall['auc_roc']:.3f}")
            with kpi2:
                st.metric("AUC-PR", f"{overall['auc_pr']:.3f}")
            with kpi3:
                st.metric("EER", f"{overall['eer']:.3f}")
            with kpi4:
                st.metric(
                    "Test Sequences",
                    f"{overall['num_sequences']:,}",
                    f"{overall['num_anomalous']:,} anomalous",
                )

            # --- Per-Camera Performance ---
            per_camera = eval_data.get("per_camera", {})
            if per_camera:
                st.markdown("---")
                st.subheader("Per-Camera Performance")

                cam_cols = st.columns(len(per_camera))
                for i, (cam_id, cam_data) in enumerate(sorted(per_camera.items())):
                    with cam_cols[i]:
                        # Status indicator
                        cam_auc = cam_data["auc_roc"]
                        status_color = "green" if cam_auc > 0.7 else "orange" if cam_auc > 0.6 else "red"
                        st.markdown(
                            f"**Camera {cam_id}** "
                            f"<span style='color:{status_color}; font-size:1.2em;'>&#9679;</span>",
                            unsafe_allow_html=True,
                        )
                        st.metric("AUC-ROC", f"{cam_data['auc_roc']:.3f}", label_visibility="collapsed")
                        st.caption(
                            f"AUC-PR: {cam_data['auc_pr']:.3f} | "
                            f"EER: {cam_data['eer']:.3f}"
                        )

                # Bar chart comparison
                fig_cam = go.Figure()
                cam_ids = sorted(per_camera.keys())
                fig_cam.add_trace(go.Bar(
                    x=[f"Camera {c}" for c in cam_ids],
                    y=[per_camera[c]["auc_roc"] for c in cam_ids],
                    name="AUC-ROC",
                    marker_color="steelblue",
                ))
                fig_cam.add_trace(go.Bar(
                    x=[f"Camera {c}" for c in cam_ids],
                    y=[per_camera[c]["auc_pr"] for c in cam_ids],
                    name="AUC-PR",
                    marker_color="coral",
                ))
                fig_cam.update_layout(
                    barmode="group",
                    height=350,
                    yaxis_title="Score",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig_cam, use_container_width=True)

            # --- Anomaly Score Timeline ---
            per_video = eval_data.get("per_video", {})
            if per_video:
                st.markdown("---")
                st.subheader("Anomaly Score Timeline")

                # Camera filter
                available_cams = sorted({v["camera_id"] for v in per_video.values()})
                selected_cam = st.selectbox(
                    "Filter by Camera",
                    ["All"] + [f"Camera {c}" for c in available_cams],
                    key="anomaly_cam_filter",
                )

                filtered_videos = per_video
                if selected_cam != "All":
                    cam_num = int(selected_cam.split()[-1])
                    filtered_videos = {
                        k: v for k, v in per_video.items()
                        if v["camera_id"] == cam_num
                    }

                # Sort by max anomaly score (most anomalous first)
                sorted_videos = sorted(
                    filtered_videos.items(),
                    key=lambda x: x[1]["max_score"],
                    reverse=True,
                )

                # Show top anomalous videos
                st.markdown(f"**Anomaly Threshold (EER):** `{threshold:.6f}`")

                # Timeline for selected videos
                top_n = min(10, len(sorted_videos))
                if top_n > 0:
                    fig_timeline = go.Figure()
                    for vid_name, vid_data in sorted_videos[:top_n]:
                        timeline = vid_data["score_timeline"]
                        color = "red" if vid_data["has_anomaly"] else "steelblue"
                        fig_timeline.add_trace(go.Scatter(
                            x=list(range(len(timeline))),
                            y=timeline,
                            mode="lines",
                            name=f"{vid_name} (Cam {vid_data['camera_id']})",
                            line=dict(color=color, width=1.5),
                            opacity=0.8,
                        ))

                    # Add threshold line
                    fig_timeline.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Threshold",
                        annotation_position="top right",
                    )
                    fig_timeline.update_layout(
                        xaxis_title="Sequence Index",
                        yaxis_title="Anomaly Score",
                        height=400,
                        legend=dict(font=dict(size=10)),
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

            # --- Alert Panel ---
            if per_video:
                st.markdown("---")
                st.subheader("Anomaly Alerts")

                anomalous_videos = [
                    (vid, data) for vid, data in sorted_videos
                    if data["max_score"] > threshold
                ]

                if anomalous_videos:
                    st.warning(f"{len(anomalous_videos)} videos with anomaly scores above threshold")

                    alert_data = []
                    for vid_name, vid_data in anomalous_videos[:20]:
                        alert_data.append({
                            "Video": vid_name,
                            "Camera": vid_data["camera_id"],
                            "Max Score": f"{vid_data['max_score']:.4f}",
                            "Mean Score": f"{vid_data['mean_score']:.4f}",
                            "Ground Truth": "Anomalous" if vid_data["has_anomaly"] else "Normal",
                            "Sequences": vid_data["num_sequences"],
                        })
                    st.dataframe(alert_data, use_container_width=True)
                else:
                    st.success("No anomalies detected above threshold")

            # --- Model Config ---
            if config:
                with st.expander("Model Configuration"):
                    st.json(config)

            # --- Privacy Note ---
            st.markdown("---")
            st.caption(
                "Privacy-by-design: This anomaly detection system operates exclusively on "
                "skeleton keypoint coordinates (body joint positions). No pixel-level image data "
                "is processed or stored, ensuring individual privacy while maintaining detection accuracy."
            )
