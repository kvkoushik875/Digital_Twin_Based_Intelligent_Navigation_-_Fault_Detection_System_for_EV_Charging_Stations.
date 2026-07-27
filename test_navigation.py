import pandas as pd
from src.navigation.gps_utils import haversine
from src.navigation.navigation_3d import navigation_map

def test_haversine():
    d = haversine(17.45, 78.53, 17.46, 78.54)
    assert d > 0
    assert d < 5  # small distance

def test_navigation_map():
    df = pd.DataFrame({
        "latitude": [17.45, 17.46],
        "longitude": [78.53, 78.54],
        "fault_type": ["normal", "overheating"]
    })

    fig = navigation_map(df, 17.45, 78.53)
    assert fig is not None
