import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df["Stage1.Output.Measurement0.U.Actual"].plot()


def clean_series(series, window_size=15, std_dev=3):
    """
    Cleans a pandas Series by removing outliers and zero values, then fills missing values.

    Parameters:
    series (pandas.Series): Series to clean.
    window_size (int): Window size for rolling calculations.
    std_dev (float): Standard deviation threshold for outliers.

    Returns:
    pandas.Series: Cleaned series.
    """
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()

    outliers = np.abs(series - rolling_mean) > (std_dev * rolling_std)

    cleaned_series = series.copy()
    cleaned_series[outliers] = np.nan

    cleaned_series[cleaned_series == 0] = np.nan

    cleaned_series.interpolate(method="linear", inplace=True)

    return cleaned_series


cleaned_series = clean_series(
    df["Stage1.Output.Measurement0.U.Actual"], window_size=100
)

cleaned_series.plot()

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_series

df["Stage1.Output.Measurement0.U.Actual"].plot()
df = df.iloc[:, :42]


def engineer_features(df, lag_features, window_size=10):
    """
    Engineers features for a DataFrame by creating lag features and rolling statistics.

    Parameters:
    df (pandas.DataFrame): DataFrame to engineer features for.
    lag_features (list): List of column names to create lag features for.
    window_size (int): Window size for rolling calculations.

    Returns:
    pandas.DataFrame: DataFrame with engineered features.
    """
    df_eng = df.copy()

    for col in lag_features:
        if col != "Stage1.Output.Measurement0.U.Actual":
            for i in range(1, window_size + 1):
                df_eng[f"{col}_lag{i}"] = df[col].shift(i)

    for col in lag_features:
        if col != "Stage1.Output.Measurement0.U.Actual":
            df_eng[f"{col}_rolling_mean"] = df[col].rolling(window=window_size).mean()
            df_eng[f"{col}_rolling_std"] = df[col].rolling(window=window_size).std()
            df_eng[f"{col}_rolling_min"] = df[col].rolling(window=window_size).min()
            df_eng[f"{col}_rolling_max"] = df[col].rolling(window=window_size).max()

    df_eng = df_eng.dropna()

    return df_eng


lag_features = df.columns.tolist()
window_size = 60

df_eng = engineer_features(df, lag_features, window_size)

df_eng.to_pickle("../../data/interim/data_engineered.pkl")
