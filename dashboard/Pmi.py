import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from copy import deepcopy
from typing import Optional

st.set_page_config(
    page_title="VisionX · Parking Analytics",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Space Mono', monospace !important;
    background: #07090f !important;
    color: #c8d6e0 !important;
}
.main, .block-container {
    background: #07090f !important;
    padding: 1.5rem 2rem 2rem 2rem !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"] { background: #07090f !important; }

.vx-header {
    display: flex; align-items: baseline; gap: 18px;
    margin-bottom: 2rem; padding-bottom: 1rem;
    border-bottom: 1px solid #1a2535;
}
.vx-logo {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 800; letter-spacing: -1px; color: #e8f0f7;
}
.vx-logo span { color: #f7a800; }
.vx-scene {
    font-size: 0.65rem; letter-spacing: 4px;
    color: #2a3a4a; text-transform: uppercase; margin-left: auto;
}
.vx-status { font-size: 0.65rem; letter-spacing: 2px; color: #22c55e; }

.metric-row {
    display: flex; gap: 1px; background: #1a2535;
    border: 1px solid #1a2535; border-radius: 6px;
    overflow: hidden; margin-bottom: 2rem; flex-wrap: wrap;
}
.metric-card {
    flex: 1 1 16%;
    min-width: 180px;
    background: #0d1320; padding: 16px 18px;
}
.metric-card .label {
    font-size: 0.55rem; letter-spacing: 3px; text-transform: uppercase;
    color: #2a4050; margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif; font-size: 1.7rem;
    font-weight: 800; color: #f7a800; line-height: 1;
}
.metric-card .value.alt { color: #06b6d4; font-size: 1.1rem; }
.metric-card .sub { font-size: 0.55rem; color: #2a4050; margin-top: 4px; }

.panel-q {
    font-size: 0.6rem; letter-spacing: 4px; text-transform: uppercase;
    color: #2a4050; margin-bottom: 2px; font-family: 'Space Mono', monospace;
}
.panel-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800;
    color: #e8f0f7; margin-bottom: 6px; padding-bottom: 8px;
    border-bottom: 1px solid #1a2535;
}
.panel-caption {
    font-size: 0.65rem; color: #2a4050; margin-bottom: 1rem; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

BASE_LAYOUT = dict(
    paper_bgcolor="#0d1320",
    plot_bgcolor="#07090f",
    font=dict(family="Space Mono", color="#c8d6e0", size=11),
    margin=dict(l=44, r=20, t=28, b=44),
    xaxis=dict(gridcolor="#131f2e", linecolor="#1a2535", zerolinecolor="#1a2535"),
    yaxis=dict(gridcolor="#131f2e", linecolor="#1a2535", zerolinecolor="#1a2535"),
    legend=dict(bgcolor="#0d1320", bordercolor="#1a2535", borderwidth=1),
)

def themed_layout(**updates):
    layout = deepcopy(BASE_LAYOUT)
    layout.update(updates)
    return layout

# ── DATA ──────────────────────────────────────────────────────────────────────
scene = "DJI_0012"

SCRIPT_DIR = Path(__file__).resolve().parent
candidate_dirs = [
    SCRIPT_DIR / "data" / "processed" / scene,
    SCRIPT_DIR.parent / "data" / "processed" / scene,
    Path(r"D:\VisionX\SpringHackathon-2026-VisionX\data\processed") / scene,
]

existing_dirs = [p for p in candidate_dirs if p.exists()]
DATA_DIR = existing_dirs[0] if existing_dirs else candidate_dirs[0]

@st.cache_data
def load_csv(filename: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / filename
    if not p.exists():
        st.error(f"File not found: {p}")
        return None
    return pd.read_csv(p)

df = load_csv("advanced_metrics.csv")

with st.expander("Debug data path", expanded=False):
    st.write("Resolved DATA_DIR:", str(DATA_DIR))
    st.write("Exists:", DATA_DIR.exists())
    if DATA_DIR.exists():
        st.write("CSV files:", [f.name for f in DATA_DIR.glob("*.csv")])

if df is None or df.empty:
    st.stop()

required_cols = {"timestamp", "passengers", "area", "frame_idx"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"advanced_metrics.csv is missing required columns: {sorted(missing_cols)}")
    st.stop()

df = df.copy()
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce").fillna(0)
df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
df["area"] = df["area"].astype(str)
df = df.dropna(subset=["timestamp", "area", "frame_idx"]).sort_values(["area", "timestamp"])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def max_zero_streak(series: pd.Series) -> int:
    max_streak = 0
    current = 0
    for v in series.to_numpy():
        if v <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)

def merge_close_surge_runs(
    timeline: pd.DataFrame,
    time_col: str,
    surge_col: str = "is_surge",
    max_gap_frames: int = 3,
) -> pd.DataFrame:
    out = timeline.sort_values(time_col).copy()
    vals = out[surge_col].to_numpy().copy()

    n = len(vals)
    i = 0
    while i < n:
        if vals[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not vals[i]:
            i += 1
        gap_end = i - 1
        gap_len = gap_end - gap_start + 1

        left_is_surge = gap_start - 1 >= 0 and vals[gap_start - 1]
        right_is_surge = i < n and vals[i]

        if left_is_surge and right_is_surge and gap_len <= max_gap_frames:
            vals[gap_start:i] = True

    out[surge_col] = vals
    return out

def add_surge_features(
    data: pd.DataFrame,
    value_col: str,
    short_window: int = 30,
    long_window: int = 300,
    sigma: float = 2.0,
    min_absolute_value: float = 0.0,
    warmup_frames: Optional[int] = None,
    run_col: str = "run_id",
) -> pd.DataFrame:
    out = data.sort_values("timestamp").copy()

    out["smooth"] = out[value_col].rolling(short_window, min_periods=1).mean()
    out["base"] = out[value_col].rolling(long_window, min_periods=1).mean()
    out["std"] = out[value_col].rolling(long_window, min_periods=1).std().fillna(0)
    out["threshold"] = out["base"] + sigma * out["std"]

    out["is_surge"] = (
        (out[value_col] > out["threshold"]) &
        (out[value_col] >= min_absolute_value)
    )

    if warmup_frames is None:
        warmup_frames = long_window

    if len(out) > 0:
        out.loc[out.index[:warmup_frames], "is_surge"] = False

    out[run_col] = (out["is_surge"] != out["is_surge"].shift(fill_value=False)).cumsum()
    return out

def build_zone_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    zone = (
        dataframe.groupby("area")
        .agg(
            avg_peds=("passengers", "mean"),
            peak_peds=("passengers", "max"),
            total_detections=("passengers", "sum"),
            inactive_ratio=("passengers", lambda s: (s <= 0).mean()),
            frames=("passengers", "size"),
        )
        .reset_index()
    )

    streaks = (
        dataframe.sort_values(["area", "timestamp"])
        .groupby("area")["passengers"]
        .apply(max_zero_streak)
        .reset_index(name="max_empty_streak")
    )

    zone = zone.merge(streaks, on="area", how="left")

    coords = zone["area"].str.extract(r"cell_(\d+)_(\d+)")
    zone["row"] = pd.to_numeric(coords[0], errors="coerce")
    zone["col"] = pd.to_numeric(coords[1], errors="coerce")

    zone["inactive_pct"] = zone["inactive_ratio"] * 100
    zone["active_pct"] = 100 - zone["inactive_pct"]
    return zone.dropna(subset=["row", "col"]).copy()

def build_scene_timeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    tl = (
        dataframe.groupby("timestamp", as_index=False)["passengers"]
        .sum()
        .rename(columns={"passengers": "scene_peds"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    short_window = 30
    long_window = 300
    warmup_frames = 150

    tl["smooth"] = tl["scene_peds"].rolling(short_window, min_periods=1).mean()
    tl["base"] = tl["scene_peds"].rolling(long_window, min_periods=1).mean()
    tl["std"] = tl["scene_peds"].rolling(long_window, min_periods=1).std().fillna(0)

    tl["enter_threshold"] = tl["base"] + 2.6 * tl["std"]
    tl["exit_threshold"] = tl["base"] + 1.4 * tl["std"]
    tl["threshold"] = tl["enter_threshold"]
    tl["min_excess"] = 2.5

    is_surge = []
    active = False

    for i, row in tl.iterrows():
        if i < warmup_frames:
            is_surge.append(False)
            continue

        smooth_val = row["smooth"]
        base_val = row["base"]
        enter_th = row["enter_threshold"]
        exit_th = row["exit_threshold"]

        can_enter = (
            (smooth_val >= enter_th) and
            ((smooth_val - base_val) >= row["min_excess"]) and
            (smooth_val >= 6)
        )

        can_stay = (
            (smooth_val >= exit_th) and
            (smooth_val >= 5)
        )

        if not active:
            active = can_enter
        else:
            active = can_stay

        is_surge.append(active)

    tl["is_surge"] = is_surge

    tl = merge_close_surge_runs(
        tl,
        time_col="timestamp",
        surge_col="is_surge",
        max_gap_frames=10,
    )

    tl["run_id"] = (tl["is_surge"] != tl["is_surge"].shift(fill_value=False)).cumsum()
    return tl

def build_zone_timeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    base = (
        dataframe.groupby(["area", "timestamp"], as_index=False)["passengers"]
        .sum()
        .rename(columns={"passengers": "zone_peds"})
        .sort_values(["area", "timestamp"])
    )

    chunks = []
    for area, g in base.groupby("area", sort=False):
        gg = add_surge_features(
            g,
            value_col="zone_peds",
            short_window=20,
            long_window=180,
            sigma=2.8,
            min_absolute_value=3,
            warmup_frames=180,
            run_col="zone_run_id",
        )

        gg = merge_close_surge_runs(
            gg,
            time_col="timestamp",
            surge_col="is_surge",
            max_gap_frames=4,
        )

        gg["zone_run_id"] = (gg["is_surge"] != gg["is_surge"].shift(fill_value=False)).cumsum()
        gg["area"] = area
        chunks.append(gg)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def build_surge_episodes(
    timeline: pd.DataFrame,
    group_cols: list,
    time_col: str,
    value_col: str,
    min_frames: int = 5,
) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame(columns=[*group_cols, "start_time", "end_time", "peak_value", "duration_frames"])

    surge_only = timeline[timeline["is_surge"]].copy()
    if surge_only.empty:
        return pd.DataFrame(columns=[*group_cols, "start_time", "end_time", "peak_value", "duration_frames"])

    episodes = (
        surge_only.groupby(group_cols)
        .agg(
            start_time=(time_col, "min"),
            end_time=(time_col, "max"),
            peak_value=(value_col, "max"),
            duration_frames=(value_col, "size"),
        )
        .reset_index()
    )

    return episodes[episodes["duration_frames"] >= min_frames].reset_index(drop=True)

def build_episode_peaks_scene(timeline: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    peaks = []
    for _, ep in episodes.iterrows():
        sub = timeline[
            (timeline["timestamp"] >= ep["start_time"]) &
            (timeline["timestamp"] <= ep["end_time"])
        ]
        if not sub.empty:
            peak_row = sub.loc[sub["scene_peds"].idxmax()]
            peaks.append({
                "timestamp": peak_row["timestamp"],
                "scene_peds": peak_row["scene_peds"],
            })
    return pd.DataFrame(peaks)

def build_zone_surge_summary(zone_timeline: pd.DataFrame, min_frames: int = 8):
    episodes = build_surge_episodes(
        zone_timeline,
        group_cols=["area", "zone_run_id"],
        time_col="timestamp",
        value_col="zone_peds",
        min_frames=min_frames,
    )

    base = (
        zone_timeline.groupby("area")
        .agg(
            zone_avg=("zone_peds", "mean"),
            zone_peak=("zone_peds", "max"),
            surge_frames=("is_surge", "sum"),
        )
        .reset_index()
    )

    if episodes.empty:
        base["surge_episodes"] = 0
        base["strongest_surge_peak"] = 0
        base["avg_surge_duration"] = 0.0
        base["longest_surge"] = 0
        base["total_surge_severity"] = 0.0
        base["max_surge_severity"] = 0.0
        return base, episodes

    summary = (
        episodes.groupby("area")
        .agg(
            surge_episodes=("zone_run_id", "size"),
            strongest_surge_peak=("peak_value", "max"),
            avg_surge_duration=("duration_frames", "mean"),
            longest_surge=("duration_frames", "max"),
        )
        .reset_index()
    )

    severity_rows = []
    for _, ep in episodes.iterrows():
        sub = zone_timeline[
            (zone_timeline["area"] == ep["area"]) &
            (zone_timeline["timestamp"] >= ep["start_time"]) &
            (zone_timeline["timestamp"] <= ep["end_time"])
        ].copy()

        if not sub.empty:
            severity = float((sub["zone_peds"] - sub["threshold"]).clip(lower=0).sum())
            severity_rows.append({
                "area": ep["area"],
                "zone_run_id": ep["zone_run_id"],
                "surge_severity": severity,
            })

    severity_df = pd.DataFrame(severity_rows)
    if severity_df.empty:
        sev_summary = pd.DataFrame({
            "area": [],
            "total_surge_severity": [],
            "max_surge_severity": [],
        })
    else:
        sev_summary = (
            severity_df.groupby("area")
            .agg(
                total_surge_severity=("surge_severity", "sum"),
                max_surge_severity=("surge_severity", "max"),
            )
            .reset_index()
        )

    out = base.merge(summary, on="area", how="left")
    out = out.merge(sev_summary, on="area", how="left")
    out = out.fillna({
        "surge_episodes": 0,
        "strongest_surge_peak": 0,
        "avg_surge_duration": 0.0,
        "longest_surge": 0,
        "total_surge_severity": 0.0,
        "max_surge_severity": 0.0,
    })

    out["surge_episodes"] = out["surge_episodes"].astype(int)
    out["strongest_surge_peak"] = out["strongest_surge_peak"].astype(int)
    out["longest_surge"] = out["longest_surge"].astype(int)
    return out, episodes

# ── BUILD DATA PRODUCTS ───────────────────────────────────────────────────────
zone_summary = build_zone_summary(df)
if zone_summary.empty:
    st.error("No valid zone coordinates were parsed from area names. Expected format like: cell_0_0")
    st.stop()

timeline = build_scene_timeline(df)
scene_surge_episodes = build_surge_episodes(
    timeline,
    group_cols=["run_id"],
    time_col="timestamp",
    value_col="scene_peds",
    min_frames=12,
)
episode_peaks = build_episode_peaks_scene(timeline, scene_surge_episodes)

zone_timeline = build_zone_timeline(df)
zone_surge_summary, zone_surge_episodes = build_zone_surge_summary(zone_timeline, min_frames=8)

zone_summary = zone_summary.merge(zone_surge_summary, on="area", how="left").fillna({
    "zone_avg": 0,
    "zone_peak": 0,
    "surge_frames": 0,
    "surge_episodes": 0,
    "strongest_surge_peak": 0,
    "avg_surge_duration": 0.0,
    "longest_surge": 0,
    "total_surge_severity": 0.0,
    "max_surge_severity": 0.0,
})

for col in ["surge_frames", "surge_episodes", "strongest_surge_peak", "longest_surge"]:
    zone_summary[col] = zone_summary[col].astype(int)

# ── CORE METRICS ──────────────────────────────────────────────────────────────
peak_scene_peds = int(timeline["scene_peds"].max()) if not timeline.empty else 0
peak_row = (
    timeline.loc[timeline["scene_peds"].idxmax()]
    if not timeline.empty else pd.Series({"timestamp": 0, "scene_peds": 0})
)

most_loaded = (
    zone_summary.sort_values("avg_peds", ascending=False).iloc[0]
    if not zone_summary.empty else pd.Series({"area": "N/A"})
)

total_frames = int(df["frame_idx"].nunique())
timestamp_min = float(df["timestamp"].min())
timestamp_max = float(df["timestamp"].max())
duration_s = max(timestamp_max - timestamp_min, 0.0)
fps_est = (total_frames / duration_s) if duration_s > 0 else 0.0
baseline_seconds = (300 / fps_est) if fps_est > 0 else None

cumulative_detections = int(df["passengers"].sum())

underused_zones = zone_summary[
    (zone_summary["inactive_ratio"] >= 0.90) |
    (zone_summary["max_empty_streak"] >= 1000)
].copy()

n_underused_zones = int(len(underused_zones))
n_scene_alert_windows = int(len(scene_surge_episodes))
zones_with_surges = int((zone_summary["surge_episodes"] > 0).sum())

top_zone_surge = (
    zone_summary.sort_values(
        ["total_surge_severity", "surge_episodes", "strongest_surge_peak"],
        ascending=False,
    ).iloc[0]
    if not zone_summary.empty else pd.Series({
        "area": "N/A",
        "surge_episodes": 0,
        "total_surge_severity": 0,
    })
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="vx-header">
  <div class="vx-logo">VISION<span>X</span></div>
  <div class="vx-scene">⬡ {scene}</div>
  <div class="vx-status">● OFFLINE ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Peak Occupancy</div>
    <div class="value">{peak_scene_peds}</div>
    <div class="sub">max simultaneous people in scene</div>
  </div>
  <div class="metric-card">
    <div class="label">Most Loaded Zone</div>
    <div class="value alt">{most_loaded["area"]}</div>
    <div class="sub">highest avg occupancy</div>
  </div>
  <div class="metric-card">
    <div class="label">Underused Zones</div>
    <div class="value">{n_underused_zones}</div>
    <div class="sub">90%+ inactive or 1000+ empty frames</div>
  </div>
  <div class="metric-card">
    <div class="label">Scene Alert Windows</div>
    <div class="value">{n_scene_alert_windows}</div>
    <div class="sub">high-activity periods above adaptive baseline</div>
  </div>
  <div class="metric-card">
    <div class="label">Zones With Surges</div>
    <div class="value">{zones_with_surges}</div>
    <div class="sub">areas with at least one surge episode</div>
  </div>
  <div class="metric-card">
    <div class="label">Most Severe Zone Surge</div>
    <div class="value alt">{top_zone_surge["area"]}</div>
    <div class="sub">{int(top_zone_surge["surge_episodes"])} episodes · severity {top_zone_surge["total_surge_severity"]:.1f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Cumulative Detections</div>
    <div class="value">{cumulative_detections:,}</div>
    <div class="sub">sum across frames · not unique people</div>
  </div>
  <div class="metric-card">
    <div class="label">Video Duration</div>
    <div class="value alt">{int(duration_s // 60)}m {int(duration_s % 60)}s</div>
    <div class="sub">{total_frames:,} unique frames processed</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 + 3 — Heatmap + Top 5 zones
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="panel-q">PANEL 1 · WHERE IS SUSTAINED LOAD?</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Zone Activity Heatmap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Each square is one area of the parking lot. '
        'Brighter means higher average pedestrians per frame across the full video. '
        'Use this to see which parts of the scene carry sustained crowd load, not just one-time spikes.</div>',
        unsafe_allow_html=True
    )

    nr = int(zone_summary["row"].max()) + 1
    nc = int(zone_summary["col"].max()) + 1
    grid = np.zeros((nr, nc))
    labels = [[""] * nc for _ in range(nr)]

    for _, r in zone_summary.iterrows():
        rr = int(r["row"])
        cc = int(r["col"])
        grid[rr, cc] = r["avg_peds"]
        labels[rr][cc] = (
            f"{r['area']}"
            f"<br>{r['avg_peds']:.2f} avg peds/frame"
            f"<br>peak {int(r['peak_peds'])}"
            f"<br>zone surges {int(r['surge_episodes'])}"
        )

    fig1 = go.Figure(go.Heatmap(
        z=grid,
        text=labels,
        colorscale=[
            [0.00, "#07090f"],
            [0.15, "#0a1f30"],
            [0.45, "#0e4a6e"],
            [0.75, "#f7a800"],
            [1.00, "#ff3a3a"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Avg peds/frame", font=dict(color="#c8d6e0", size=10)),
            tickfont=dict(color="#c8d6e0"),
            bgcolor="#0d1320",
            bordercolor="#1a2535",
            borderwidth=1,
        ),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig1.update_layout(**themed_layout(
        height=360,
        xaxis=dict(
            title="→ Column",
            tickvals=list(range(nc)),
            ticktext=[f"col {i}" for i in range(nc)],
            gridcolor="#131f2e",
            linecolor="#1a2535",
            zerolinecolor="#1a2535",
        ),
        yaxis=dict(
            title="Row ↑",
            tickvals=list(range(nr)),
            ticktext=[f"row {i}" for i in range(nr)],
            gridcolor="#131f2e",
            linecolor="#1a2535",
            zerolinecolor="#1a2535",
        ),
    ))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown('<div class="panel-q">PANEL 3 · WHICH ZONES NEED ATTENTION?</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Top 5 Monitoring Priority Zones</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">These zones are ranked by average pedestrians per frame. '
        'This keeps the dashboard consistent with the heatmap and shows where continuous monitoring matters most.</div>',
        unsafe_allow_html=True
    )

    top5 = zone_summary.nlargest(5, "avg_peds").sort_values("avg_peds")
    bar_colors = ["#132a3a", "#1a3f55", "#0e5a7a", "#06b6d4", "#f7a800"]

    fig3 = go.Figure(go.Bar(
        x=top5["avg_peds"],
        y=top5["area"],
        orientation="h",
        marker=dict(color=bar_colors[-len(top5):]),
        text=[f"{v:.2f} avg" for v in top5["avg_peds"]],
        textposition="outside",
        textfont=dict(color="#c8d6e0", size=10),
        hovertemplate="%{y}: %{x:.2f} avg peds/frame<extra></extra>",
    ))
    fig3.update_layout(**themed_layout(
        height=360,
        showlegend=False,
        xaxis=dict(
            title="Average pedestrians per frame",
            gridcolor="#131f2e",
            linecolor="#1a2535",
            zerolinecolor="#1a2535",
        ),
        yaxis=dict(
            gridcolor="#131f2e",
            linecolor="#1a2535",
            zerolinecolor="#1a2535",
        ),
    ))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Scene occupancy over time
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="panel-q">PANEL 2 · WHEN IS THE WHOLE SCENE BUSY?</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Total Pedestrians Over Time</div>', unsafe_allow_html=True)

baseline_caption = (
    f'The dashed orange baseline uses a 300-frame rolling window '
    f'(about {baseline_seconds:.1f}s at the estimated frame rate).'
    if baseline_seconds is not None else
    'The dashed orange baseline uses a 300-frame rolling window.'
)

st.markdown(
    f'<div class="panel-caption">One line represents total scene occupancy. '
    f'The cyan line shows the raw count, the orange line shows the rolling baseline, '
    f'and the white line shows the short-term average. {baseline_caption}</div>',
    unsafe_allow_html=True
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["scene_peds"],
    mode="lines",
    name="Raw count",
    line=dict(color="#06b6d4", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(6,182,212,0.10)",
    hovertemplate="t=%{x:.1f}s · %{y} peds<extra></extra>",
))
fig2.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["base"],
    mode="lines",
    name="300-frame baseline",
    line=dict(color="#f7a800", width=2.2, dash="dash"),
    hovertemplate="baseline=%{y:.1f}<extra></extra>",
))
fig2.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["smooth"],
    mode="lines",
    name="30-frame avg",
    line=dict(color="#c8d6e0", width=1.2),
    hovertemplate="avg=%{y:.1f}<extra></extra>",
))
fig2.add_vline(
    x=peak_row["timestamp"],
    line_dash="dash",
    line_color="#ff4444",
    line_width=1.5,
    annotation_text=f"PEAK: {int(peak_row['scene_peds'])} @ {peak_row['timestamp']:.0f}s",
    annotation_font=dict(color="#ff4444", size=10),
    annotation_position="top right",
)
fig2.update_layout(**themed_layout(
    height=290,
    xaxis_title="Timestamp (s)",
    yaxis_title="Pedestrians in scene",
    legend=dict(
        bgcolor="#0d1320",
        bordercolor="#1a2535",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
    ),
))
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Underused zones
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="panel-q">PANEL 4 · WHICH AREAS ARE UNDERUSED?</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Underused Zone Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-caption">This panel shows zones with the highest inactivity. '
    'Use it to spot wasted space, weak routing, blocked access, or areas people consistently avoid.</div>',
    unsafe_allow_html=True
)

col_dz1, col_dz2 = st.columns([3, 2], gap="large")

with col_dz1:
    underused_rank = (
        zone_summary.sort_values(["inactive_ratio", "max_empty_streak"], ascending=[False, False])
        .head(10)
        .sort_values("inactive_ratio")
    )

    fig4 = go.Figure(go.Bar(
        x=underused_rank["inactive_pct"],
        y=underused_rank["area"],
        orientation="h",
        marker=dict(
            color=underused_rank["inactive_pct"],
            colorscale=[
                [0.00, "#0e4a6e"],
                [0.60, "#06b6d4"],
                [0.85, "#f7a800"],
                [1.00, "#ff3a3a"],
            ],
            cmin=0,
            cmax=100,
            colorbar=dict(
                title=dict(text="Inactive %", font=dict(color="#c8d6e0", size=10)),
                tickfont=dict(color="#c8d6e0"),
                bgcolor="#0d1320",
                bordercolor="#1a2535",
                borderwidth=1,
            ),
        ),
        text=[f"{v:.1f}%" for v in underused_rank["inactive_pct"]],
        textposition="outside",
        hovertemplate="%{y}<br>inactive %{x:.1f}%<extra></extra>",
    ))
    fig4.update_layout(**themed_layout(
        height=380,
        showlegend=False,
        xaxis_title="Inactive ratio (%)",
        yaxis_title="",
    ))
    st.plotly_chart(fig4, use_container_width=True)

with col_dz2:
    st.markdown("**Most underused zones:**")
    show = (
        zone_summary.sort_values(["inactive_ratio", "max_empty_streak"], ascending=[False, False])
        .head(10)[["area", "inactive_pct", "max_empty_streak", "avg_peds", "peak_peds"]]
        .reset_index(drop=True)
    )
    show.columns = ["Zone", "Inactive %", "Longest Empty Streak", "Avg Peds/Frame", "Peak Peds"]

    st.dataframe(
        show.style
            .background_gradient(subset=["Inactive %"], cmap="YlOrRd")
            .format({
                "Inactive %": "{:.1f}",
                "Avg Peds/Frame": "{:.2f}",
                "Peak Peds": "{:.0f}",
                "Longest Empty Streak": "{:.0f}",
            }),
        use_container_width=True,
        height=380,
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Scene alert windows
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="panel-q">PANEL 5 · WHEN DOES ACTIVITY BECOME ABNORMAL?</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Scene High-Activity Windows</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-caption">The dashed orange line is the rolling baseline and the red line is the scene entry threshold. '
    'Red triangles mark the peak of each high-activity window. Scene detection uses hysteresis on the smoothed signal so small bumps are not over-counted.</div>',
    unsafe_allow_html=True
)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["scene_peds"],
    mode="lines",
    name="Total peds",
    line=dict(color="#06b6d4", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(6,182,212,0.06)",
    hovertemplate="t=%{x:.1f}s · %{y} peds<extra></extra>",
))
fig5.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["base"],
    mode="lines",
    name="300-frame baseline",
    line=dict(color="#f7a800", width=2, dash="dash"),
    hovertemplate="baseline=%{y:.1f}<extra></extra>",
))
fig5.add_trace(go.Scatter(
    x=timeline["timestamp"],
    y=timeline["threshold"],
    mode="lines",
    name="Entry threshold",
    line=dict(color="#ff4444", width=1.3, dash="dot"),
    hovertemplate="threshold=%{y:.1f}<extra></extra>",
))
if not episode_peaks.empty:
    fig5.add_trace(go.Scatter(
        x=episode_peaks["timestamp"],
        y=episode_peaks["scene_peds"],
        mode="markers",
        name=f"Window peaks ({n_scene_alert_windows})",
        marker=dict(symbol="triangle-up", size=8, color="#ff3a3a"),
        hovertemplate="⚠ scene window peak t=%{x:.1f}s · %{y} peds<extra></extra>",
    ))
fig5.update_layout(**themed_layout(
    height=300,
    xaxis_title="Timestamp (s)",
    yaxis_title="Total pedestrians in scene",
    legend=dict(
        bgcolor="#0d1320",
        bordercolor="#1a2535",
        borderwidth=1,
        orientation="h",
        yanchor="bottom",
        y=1.02,
    ),
))
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Zone surge hotspots
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="panel-q">PANEL 6 · WHICH ZONES SURGE ABNORMALLY?</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Zone Surge Hotspots</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-caption">This panel adds surge detection per zone, not just for the whole scene. '
    'Zone surge is stricter than scene surge: it uses warmup suppression, a higher threshold, '
    'a minimum occupancy floor, and merged nearby bursts so noise does not create fake episodes.</div>',
    unsafe_allow_html=True
)

col_sz1, col_sz2 = st.columns([3, 2], gap="large")

zone_surge_rank_base = zone_summary[zone_summary["surge_episodes"] > 0].copy()
if zone_surge_rank_base.empty:
    zone_surge_rank_base = zone_summary.copy()

zone_surge_rank = (
    zone_surge_rank_base.sort_values(
        ["total_surge_severity", "surge_episodes", "strongest_surge_peak"],
        ascending=[False, False, False],
    )
    .head(10)
    .sort_values("total_surge_severity")
)

with col_sz1:
    fig6 = go.Figure(go.Bar(
        x=zone_surge_rank["total_surge_severity"],
        y=zone_surge_rank["area"],
        orientation="h",
        marker=dict(
            color=zone_surge_rank["strongest_surge_peak"],
            colorscale=[
                [0.00, "#0e4a6e"],
                [0.60, "#06b6d4"],
                [0.85, "#f7a800"],
                [1.00, "#ff3a3a"],
            ],
            colorbar=dict(
                title=dict(text="Strongest peak", font=dict(color="#c8d6e0", size=10)),
                tickfont=dict(color="#c8d6e0"),
                bgcolor="#0d1320",
                bordercolor="#1a2535",
                borderwidth=1,
            ),
        ),
        text=[
            f"{int(ep)} eps · peak {int(pk)}"
            for ep, pk in zip(zone_surge_rank["surge_episodes"], zone_surge_rank["strongest_surge_peak"])
        ],
        textposition="outside",
        hovertemplate="%{y}<br>total severity %{x:.1f}<extra></extra>",
    ))
    fig6.update_layout(**themed_layout(
        height=380,
        showlegend=False,
        xaxis_title="Total surge severity",
        yaxis_title="",
    ))
    st.plotly_chart(fig6, use_container_width=True)

with col_sz2:
    st.markdown("**Zone surge summary:**")
    zone_surge_table = (
        zone_surge_rank.sort_values(
            ["total_surge_severity", "surge_episodes", "strongest_surge_peak"],
            ascending=[False, False, False],
        )[["area", "surge_episodes", "strongest_surge_peak", "avg_surge_duration", "total_surge_severity", "avg_peds"]]
        .reset_index(drop=True)
    )
    zone_surge_table.columns = [
        "Zone",
        "Surge Episodes",
        "Strongest Peak",
        "Avg Surge Duration",
        "Total Surge Severity",
        "Avg Peds/Frame",
    ]

    st.dataframe(
        zone_surge_table.style.format({
            "Surge Episodes": "{:.0f}",
            "Strongest Peak": "{:.0f}",
            "Avg Surge Duration": "{:.1f}",
            "Total Surge Severity": "{:.1f}",
            "Avg Peds/Frame": "{:.2f}",
        }),
        use_container_width=True,
        height=380,
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="font-size:0.55rem;color:#1a2535;letter-spacing:3px;text-align:center;padding:8px">'
    'VISIONX · SPRING HACKATHON 2026 · AERIAL PARKING INTELLIGENCE SYSTEM'
    '</div>',
    unsafe_allow_html=True
)