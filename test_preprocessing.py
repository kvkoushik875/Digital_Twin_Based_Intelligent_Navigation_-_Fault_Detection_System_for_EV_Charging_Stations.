import pandas as pd
import numpy as np
from src.preprocessing.clean_data import clean_and_scale
from src.preprocessing.label_faults import label_faults

def test_clean_and_scale():
    df = pd.DataFrame({
        "voltage": [400, 410, 420],
        "current": [30, 32, 33],
        "temperature": [45, 46, 47],
        "power": [12, 13, 14],
        "comm_status": [1, 1, 1]
    })

    df_clean, scaled, scaler = clean_and_scale(df)

    assert scaled.shape == (3, 5)
    assert df_clean["voltage"].iloc[1] != 410  # rolling applied

def test_label_faults():
    df = pd.DataFrame({
        "voltage": [400, 450],  # big jump → voltage fault
        "current": [30, 32],
        "temperature": [45, 46],
        "power": [12, 13],
        "comm_status": [1, 1]
    })

    df_labeled = label_faults(df)
    assert df_labeled["fault_type"].iloc[1] == "voltage_fluctuation"
