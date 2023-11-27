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
    This function creates line plots for the specified columns of a DataFrame.

    It iterates over the provided columns and for each column, it creates a line plot 
    using seaborn's lineplot function. The x-axis represents the DataFrame's index 
    (assumed to be time in this context), and the y-axis represents the values of the column.

    Each plot is labeled with the column's name and a legend is created. 
    The plots are displayed with a specified title.

    Parameters:
    df (pandas.DataFrame): The DataFrame to plot.
    columns (list): The list of column names to create plots for.
    title (str): The title for the plots.
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
