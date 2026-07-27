import plotly.graph_objects as go

# FIXED IMPORT
from src.navigation.gps_utils import haversine


def navigation_map(df, user_lat, user_lon):
    df = df.copy()

    # compute distance to each station
    df["distance"] = df.apply(
        lambda r: haversine(user_lat, user_lon, r["latitude"], r["longitude"]),
        axis=1
    )

    # nearest healthy station
    target = df[df["fault_type"] == "normal"].sort_values("distance").iloc[0]

    fig = go.Figure()

    # all stations
    fig.add_trace(go.Scatter3d(
        x=df["longitude"],
        y=df["latitude"],
        z=[0] * len(df),
        mode="markers",
        marker=dict(color=df["distance"], size=5),
        name="Stations"
    ))

    # user location
    fig.add_trace(go.Scatter3d(
        x=[user_lon],
        y=[user_lat],
        z=[0],
        mode="markers+text",
        marker=dict(size=8, color="blue"),
        text=["You"]
    ))

    # nearest healthy station
    fig.add_trace(go.Scatter3d(
        x=[target["longitude"]],
        y=[target["latitude"]],
        z=[0],
        mode="markers+text",
        marker=dict(size=8, color="green"),
        text=["Nearest Healthy"]
    ))

    fig.update_layout(title="3D Navigation Map")
    return fig
