"""
Virtual twin rendering for the parking lot.

Draws a top-down 2D map of all 364 parking spaces colored by occupancy status.
"""

import pandas as pd
import plotly.graph_objects as go


def render_parking_map(
    parking_spaces: pd.DataFrame,
    occupied_space_ids: set | list,
    title: str = "",
) -> go.Figure:
    """
    Render a top-down parking lot map with spaces colored by occupancy.

    Args:
        parking_spaces: DataFrame with columns id, area, and 4 corner coordinate pairs.
        occupied_space_ids: Collection of space IDs that are currently occupied.
        title: Optional title for the figure.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    occupied_set = set(int(sid) for sid in occupied_space_ids)
    n_occupied = len(occupied_set)
    total = len(parking_spaces)

    fig = go.Figure()

    # Collect hover data via invisible scatter traces at space centers
    free_x, free_y, free_text = [], [], []
    occ_x, occ_y, occ_text = [], [], []

    for _, row in parking_spaces.iterrows():
        space_id = int(row["id"])
        area = row["area"]
        is_occupied = space_id in occupied_set

        # Corner coordinates
        corners_x = [row["top_left_x"], row["top_right_x"],
                      row["btm_right_x"], row["btm_left_x"]]
        corners_y = [row["top_left_y"], row["top_right_y"],
                      row["btm_right_y"], row["btm_left_y"]]

        # SVG path for the quadrilateral
        path = (
            f"M {corners_x[0]},{corners_y[0]} "
            f"L {corners_x[1]},{corners_y[1]} "
            f"L {corners_x[2]},{corners_y[2]} "
            f"L {corners_x[3]},{corners_y[3]} Z"
        )

        color = "rgba(220, 20, 60, 0.7)" if is_occupied else "rgba(34, 139, 34, 0.6)"
        line_color = "rgba(180, 10, 40, 1)" if is_occupied else "rgba(20, 100, 20, 1)"

        fig.add_shape(
            type="path",
            path=path,
            fillcolor=color,
            line=dict(color=line_color, width=0.5),
        )

        # Center for hover
        cx = sum(corners_x) / 4
        cy = sum(corners_y) / 4
        status = "Occupied" if is_occupied else "Free"
        text = f"Space {space_id} (Area {area})<br>Status: {status}"

        if is_occupied:
            occ_x.append(cx)
            occ_y.append(cy)
            occ_text.append(text)
        else:
            free_x.append(cx)
            free_y.append(cy)
            free_text.append(text)

    # Add invisible scatter traces for hover tooltips
    fig.add_trace(go.Scatter(
        x=free_x, y=free_y, mode="markers",
        marker=dict(size=6, color="rgba(34, 139, 34, 0.01)"),
        hovertext=free_text, hoverinfo="text",
        name="Free", showlegend=True,
        legendgroup="free",
    ))
    fig.add_trace(go.Scatter(
        x=occ_x, y=occ_y, mode="markers",
        marker=dict(size=6, color="rgba(220, 20, 60, 0.01)"),
        hovertext=occ_text, hoverinfo="text",
        name="Occupied", showlegend=True,
        legendgroup="occupied",
    ))

    # Entrance zone
    fig.add_shape(
        type="rect",
        x0=5, y0=70, x1=25, y1=80,
        line=dict(color="dodgerblue", width=2, dash="dash"),
        fillcolor="rgba(30, 144, 255, 0.1)",
    )
    fig.add_annotation(
        x=15, y=75, text="Entrance", showarrow=False,
        font=dict(size=11, color="dodgerblue"),
    )

    # Summary annotation
    occ_pct = round(n_occupied / total * 100, 1) if total else 0
    summary = f"Occupied: {n_occupied}/{total} ({occ_pct}%)"
    fig.add_annotation(
        x=70, y=-5, text=summary, showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.6)", borderpad=4,
    )

    fig.update_layout(
        title=title or "Parking Lot — Virtual Twin",
        xaxis=dict(range=[-2, 142], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(
            range=[-8, 84], showgrid=False, zeroline=False, visible=False,
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="rgb(40, 40, 40)",
        paper_bgcolor="rgb(30, 30, 30)",
        font=dict(color="white"),
        height=500,
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=12),
        ),
    )

    return fig
