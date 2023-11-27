import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("../../data/interim/data_processed.pkl")

# Group columns by machine
ambient_columns = [
    col for col in df.columns if col.startswith('AmbientConditions')]
machine_columns = [col for col in df.columns if 'Machine' in col]
stage1_output_columns = [col for col in df.columns if 'Stage1.Output' in col]


def plot_columns(columns, title):
    """
    Creates line plots for specified DataFrame columns.

    Iterates over columns, creating a line plot for each using seaborn's lineplot function. 
    The x-axis is the DataFrame's index (assumed to be time), and the y-axis is the column values. 
    Each plot is labeled with the column's name and a title is displayed.

    Parameters:
    df (pandas.DataFrame): DataFrame to plot.
    columns (list): Column names to plot.
    title (str): Title for the plots.
    """
    plt.figure(figsize=(20, 5))
    for column in columns:
        sns.lineplot(data=df, x=df.index, y=column, label=column)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc="best")
    plt.show()


# Plot for each group
plot_columns(ambient_columns, 'Ambient Conditions')
plot_columns(df, machine_columns, 'Machine Columns')
plot_columns(df, stage1_output_columns, 'Stage1 Output Columns')
