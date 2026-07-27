import pandas as pd
from src.ingestion.load_data import load_dataset

def test_load_dataset():
    # Create a temporary CSV
    df_mock = pd.DataFrame({
        "voltage": [400, 410],
        "current": [30, 32],
        "temperature": [45, 46],
        "power": [12, 13],
        "comm_status": [1, 1]
    })
    df_mock.to_csv("temp_test.csv", index=False)

    df = load_dataset("temp_test.csv")

    assert not df.isna().any().any()
    assert "voltage" in df.columns
    assert len(df) == 2
