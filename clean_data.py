import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_and_scale(df):
    # Detect numeric columns automatically
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    print("Numeric columns detected:", numeric_cols)

    # Rolling smoothing only on numeric columns
    for col in numeric_cols:
        df[col] = df[col].rolling(5, min_periods=1).mean()

    # Scale numeric columns
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])

    return df, scaled, scaler
