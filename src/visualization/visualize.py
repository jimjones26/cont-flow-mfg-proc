import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("../../data/interim/data_processed.pkl")

# Group columns by machine
ambient_columns = [
    col for col in df.columns if col.startswith('AmbientConditions')]
machine_columns = [col for col in df.columns if col.startswith('Machine')]
combiner_columns = [col for col in df.columns if col.startswith(
    'FirstStage.CombinerOperation')]
stage_output_columns = [col for col in df.columns if 'Stage1.Output' in col]


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

# function to extract properties from machine columns


def get_machine_properties(machine_columns):
    properties = set()
    for col in machine_columns:
        prop = ".".join(col.split(".")[1:])
        properties.add(prop)
    return properties


machine_properties = get_machine_properties(machine_columns)


def plot_machine_columns(properties, title):
    for prop in properties:
        plt.figure(figsize=(15, 6))
        for machine in range(1, 4):
            col = f'Machine{machine}.{prop}'
            sns.lineplot(data=df, x=df.index, y=col, label=col)

        plt.title(f'{title}: {prop}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc="best")
        plt.show()

# function to create indivisual line plots for each column


def plot_individual_columns(columns, title):
    for column in columns:
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=df, x=df.index, y=column, label=column)

        plt.title(f'{title}: {column}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc="best")
        plt.show()


# Plot for each group
plot_columns(ambient_columns, 'Ambient Conditions')
plot_machine_columns(machine_properties, 'Machine Properties')
plot_columns(combiner_columns, 'First Stage Combiner Operation')
plot_individual_columns(stage_output_columns, 'Stage1 Output Measurement')
