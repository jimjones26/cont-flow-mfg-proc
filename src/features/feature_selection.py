import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression

df = pd.read_pickle("../../data/interim/data_engineered.pkl")


def select_best_features(df, target_column, k=10):
    """
    Selects the best features in relation to the target column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to select features from.
    target_column (str): The name of the target column.
    k (int): The number of best features to select.

    Returns:
    list: The names of the best features.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)

    top_features_indicies = selector.get_support(indices=True)

    best_features = X.iloc[:, top_features_indicies]

    return best_features


best_features = select_best_features(df, "Stage1.Output.Measurement0.U.Actual", k=10)
best_features.columns

best_features.to_pickle("../../data/processed/best_features.pkl")

best_features_df = pd.concat(
    [best_features, df["Stage1.Output.Measurement0.U.Actual"]], axis=1
)
