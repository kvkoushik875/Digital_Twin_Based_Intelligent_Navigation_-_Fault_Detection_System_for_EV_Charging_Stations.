import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def live_sensor_charts(df):
    """
    Draw live charts for voltage, current, temperature, power.
    df: DataFrame containing sensor data
    """

    st.markdown("### 📊 Live Sensor Charts")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=df["voltage"],
        mode="lines",
        name="Voltage (V)"
    ))

    fig.add_trace(go.Scatter(
        y=df["current"],
        mode="lines",
        name="Current (A)"
    ))

    fig.add_trace(go.Scatter(
        y=df["temperature"],
        mode="lines",
        name="Temperature (°C)"
    ))

    fig.add_trace(go.Scatter(
        y=df["power"],
        mode="lines",
        name="Power (kW)"
    ))

    fig.update_layout(
        title="Live Sensor Trends",
        xaxis_title="Time Index",
        yaxis_title="Sensor Values",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)
