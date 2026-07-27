import plotly.graph_objects as go

def digital_twin_view(fault, row):
    color_map = {
        "normal": "green",
        "voltage_fluctuation": "yellow",
        "power_loss": "red",
        "overheating": "orange",
        "communication_failure": "blue"
    }

    fig = go.Figure(data=[
        go.Scatter3d(
            x=[0,1,2],
            y=[0,1,0],
            z=[0,0,1],
            mode="markers+text",
            marker=dict(size=12, color=color_map[fault]),
            text=["Charger","Cable","Cooling"],
            textposition="top center"
        )
    ])

    fig.update_layout(title=f"Digital Twin – Fault: {fault}")
    return fig
