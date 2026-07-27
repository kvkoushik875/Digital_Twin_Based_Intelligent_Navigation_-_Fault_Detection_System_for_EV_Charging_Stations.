import pandas as pd
from src.digital_twin.twin_3d import digital_twin_view

def test_digital_twin_view():
    row = pd.Series({
        "voltage": 410,
        "current": 32,
        "temperature": 45,
        "power": 12,
        "comm_status": 1
    })

    fig = digital_twin_view("normal", row)
    assert fig is not None
