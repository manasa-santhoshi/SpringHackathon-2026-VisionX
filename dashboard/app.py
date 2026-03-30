"""
Parking Lot Analytics Dashboard.

Two tabs:
1. Vehicle Analytics — metrics from the detection/tracking pipeline (DLP)
2. Anomaly Detection — multi-camera skeleton-based anomaly detection (CHAD)

Usage:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

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
tab_vehicle, tab_anomaly = st.tabs(["Vehicle Analytics", "Anomaly Detection"])

# ============================================================
# TAB 1: Vehicle Analytics (existing functionality)
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
# TAB 2: Anomaly Detection (CHAD multi-camera)
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
