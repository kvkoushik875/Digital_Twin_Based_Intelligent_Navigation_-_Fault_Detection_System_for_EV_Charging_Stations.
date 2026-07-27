import time
import torch
import streamlit as st
import pandas as pd
import numpy as np

from src.digital_twin.twin_3d import digital_twin_view

def realtime_dashboard(model, le, scaler, mqtt_client):
    """
    Real-time dashboard for MQTT streaming + PyTorch inference.
    """

    st.markdown("### 🔴 Real-Time Fault Detection Dashboard")

    placeholder = st.empty()

    while True:
        msg = mqtt_client.get_latest()

        if msg is not None:
            row = pd.Series(msg)

            # Extract features
            features = ["voltage", "current", "temperature", "power", "comm_status"]
            x_raw = row[features].values.reshape(1, -1)
            x_scaled = scaler.transform(x_raw)

            # Create window (repeat same reading)
            window = np.repeat(x_scaled, 60, axis=0).reshape(1, 60, -1)
            sample = torch.tensor(window, dtype=torch.float32)

            with torch.no_grad():
                pred = model(sample)
                pred_class = torch.argmax(pred, dim=1).item()
                fault_label = le.inverse_transform([pred_class])[0]

            with placeholder.container():
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📡 Latest MQTT Sensor Data")
                    st.json({
                        **msg,
                        "model_predicted_fault": fault_label
                    })

                with col2:
                    st.markdown("### 🧊 Digital Twin (Live)")
                    fig = digital_twin_view(fault_label, row)
                    st.plotly_chart(fig, use_container_width=True)

        time.sleep(1)
