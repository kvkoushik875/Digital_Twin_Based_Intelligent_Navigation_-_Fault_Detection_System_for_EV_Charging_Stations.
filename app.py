import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys

# ------------------------------------------------------------
# Compute project root (where app.py is located)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------
# Correct imports
# ------------------------------------------------------------
from src.ingestion.load_data import load_dataset
from src.ingestion.mqtt_client import MQTTIngestor

from src.preprocessing.clean_data import clean_and_scale
from src.preprocessing.label_faults import label_faults

from src.modeling.window_generator import create_windows
from src.modeling.train_model import train_pytorch_model
from src.modeling.lstm_model import FaultLSTM

from src.digital_twin.twin_3d import digital_twin_view
from src.navigation.navigation_3d import navigation_map

from src.dashboard.charts import live_sensor_charts
from src.dashboard.realtime import realtime_dashboard

from src.utils.helper import load_settings
from src.utils.logger import system_logger


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
st.set_page_config(page_title="EV Digital Twin Fault Detection", layout="wide")


def main():
    st.title("🔌 EV Charging Station Digital Twin – Fault Detection")

    # ------------------------------------------------------------
    # Load settings.yaml
    # ------------------------------------------------------------
    settings = load_settings()
    system_logger.info("Settings loaded successfully.")

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "ev_charging_full_sensor_dataset.csv")

    st.write("📂 Dataset path:", data_path)

    df = load_dataset(data_path)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    df_clean, scaled, scaler = clean_and_scale(df)
    df_labeled = label_faults(df_clean)

    st.subheader("Fault Distribution")
    st.bar_chart(df_labeled["fault_type"].value_counts())

    # ------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------
    X, y = create_windows(scaled, df_labeled["fault_type"].values)

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------
    if st.button("Train PyTorch Model"):
        model, le = train_pytorch_model(X, y)

        model_path = os.path.join(PROJECT_ROOT, "data", "models", "pytorch_fault_detector.pt")
        torch.save(model.state_dict(), model_path)

        st.session_state["model"] = model
        st.session_state["le"] = le
        st.session_state["scaler"] = scaler

        st.success(f"Model trained and saved at {model_path}")

    # ------------------------------------------------------------
    # Require trained model
    # ------------------------------------------------------------
    if "model" not in st.session_state:
        st.info("Train the model first to enable real-time inference.")
        return

    model = st.session_state["model"]
    le = st.session_state["le"]
    scaler = st.session_state["scaler"]

    # ------------------------------------------------------------
    # Live sensor charts
    # ------------------------------------------------------------
    live_sensor_charts(df_clean)

    # ------------------------------------------------------------
    # MQTT Setup
    # ------------------------------------------------------------
    st.subheader("📡 Real-Time MQTT Streaming")

    broker = st.text_input("MQTT Broker Host", "localhost")
    port = st.number_input("MQTT Broker Port", 1883)
    topic = st.text_input("MQTT Topic", "ev/charging/sensors")

    if "mqtt" not in st.session_state:
        st.session_state["mqtt"] = MQTTIngestor(
            broker_host=broker,
            broker_port=port,
            topic=topic
        )
        st.session_state["mqtt"].start()

    mqtt_client = st.session_state["mqtt"]

    if st.button("Start Real-Time Dashboard"):
        realtime_dashboard(model, le, scaler, mqtt_client)

    # ------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------
    st.subheader("🧭 Navigation to Nearest Healthy Station")

    user_lat = st.number_input("User Latitude", value=17.45)
    user_lon = st.number_input("User Longitude", value=78.53)

    nav_fig = navigation_map(df_labeled, user_lat, user_lon)
    st.plotly_chart(nav_fig)


if __name__ == "__main__":
    main()
