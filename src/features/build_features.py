import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df["Stage1.Output.Measurement0.U.Actual"].plot()


def clean_series(series, window_size=15, std_dev=3):
    """
    Removes outliers from a pandas Series and fills missing values with linear interpolation.

    Parameters:
    series (pandas.Series): The series to clean.
    window_size (int): The size of the window for the rolling mean and standard deviation calculations.
    std_dev (float): The number of standard deviations from the mean to consider an outlier.

    Returns:
    pandas.Series: The cleaned series.
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
    df["Stage1.Output.Measurement0.U.Actual"], window_size=100)

cleaned_series.plot()

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_series

df["Stage1.Output.Measurement0.U.Actual"].plot()
