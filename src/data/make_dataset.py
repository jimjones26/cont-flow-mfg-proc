import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/raw/continuous_factory_process.csv')


def process_dataframe(df):
  """
    This function processes a given DataFrame by performing several operations:
    
    1. Selects the first 71 columns of the DataFrame.
    2. Removes any columns that contain the word 'Setpoint' in their name.
    3. Converts the 'time_stamp' column to datetime format.
    4. Sets the 'time_stamp' column as the index of the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame to be processed.
    
    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    selected_columns = df.columns[:71]
    df = df[selected_columns]
    # get rid of columns that contain "Setpoint" in the name
    df = df.loc[:, ~df.columns.str.contains('Setpoint')]

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp")

    return df


processed_df = process_dataframe(df)

processed_df.info()

processed_df["Machine1.RawMaterial.Property2"]

processed_df.to_pickle("../../data/interim/data_processed.pkl")