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
