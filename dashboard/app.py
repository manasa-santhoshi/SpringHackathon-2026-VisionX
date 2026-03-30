"""
Parking Lot Analytics Dashboard.

Displays metrics computed by the pipeline: vehicle count, occupancy, dwell times, entry/exit.

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

# --- Sidebar ---
scenes = get_available_scenes()
if not scenes:
    st.error(
        "No processed data found. Run the pipeline first:\n\n"
        "```\npython -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV\n```"
    )
    st.stop()

scene = st.sidebar.selectbox("Scene", scenes)
scene_dir = PROCESSED_DIR / scene

# --- Load data ---
vehicle_count = load_json(scene_dir / "vehicle_count.json")
occupancy = load_json(scene_dir / "occupancy_timeline.json")
dwell = load_json(scene_dir / "dwell_times.json")
entry_exit = load_json(scene_dir / "entry_exit.json")
baseline_occupancy = load_json(scene_dir / "baseline_occupancy.json")
gt_occupancy = load_json(scene_dir / "gt_occupancy.json")

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

    occ_opts_col1, occ_opts_col2, occ_opts_col3 = st.columns(3)
    with occ_opts_col1:
        show_areas = st.checkbox("Show by area", value=False)
    with occ_opts_col2:
        show_baseline = st.checkbox("Show baseline (frame-diff)", value=False, disabled=baseline_occupancy is None)
    with occ_opts_col3:
        show_gt = st.checkbox("Show ground truth", value=False, disabled=gt_occupancy is None)

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

    if show_gt and gt_occupancy:
        fig.add_trace(go.Scatter(
            x=gt_occupancy["timestamps"],
            y=gt_occupancy["occupied"],
            mode="lines",
            name="Ground Truth",
            line=dict(color="gold", width=2),
        ))

    if show_baseline and baseline_occupancy:
        fig.add_trace(go.Scatter(
            x=baseline_occupancy["timestamps"],
            y=baseline_occupancy["occupied"],
            mode="lines",
            name="Baseline (frame-diff)",
            line=dict(color="gray", dash="dash"),
        ))

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
