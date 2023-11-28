import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_pickle("../../data/processed/best_features.pkl")

df.info()

""" 0   Machine1.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 1   Machine1.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 2   Machine1.MaterialPressure.U.Actual_rolling_min            14028 non-null  float64
 3   Machine1.MaterialPressure.U.Actual_rolling_max            14028 non-null  float64
 4   Machine2.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 5   Machine2.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 6   Machine3.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 7   Machine3.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 8   Machine3.MotorAmperage.U.Actual_rolling_min               14028 non-null  float64
 9   Machine3.MotorAmperage.U.Actual_rolling_max               14028 non-null  float64
 10  Stage1.Output.Measurement0.U.Actual                       14028 non-null  float640   Machine1.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 1   Machine1.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 2   Machine1.MaterialPressure.U.Actual_rolling_min            14028 non-null  float64
 3   Machine1.MaterialPressure.U.Actual_rolling_max            14028 non-null  float64
 4   Machine2.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 5   Machine2.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 6   Machine3.RawMaterialFeederParameter.U.Actual_rolling_min  14028 non-null  float64
 7   Machine3.RawMaterialFeederParameter.U.Actual_rolling_max  14028 non-null  float64
 8   Machine3.MotorAmperage.U.Actual_rolling_min               14028 non-null  float64
 9   Machine3.MotorAmperage.U.Actual_rolling_max               14028 non-null  float64
 10  Stage1.Output.Measurement0.U.Actual                       14028 non-null  float64 """


def experiment_models(df, target_column, test_size=0.2, random_state=42):
    """
    Experiments with different ML models to predict a target variable.
    Splits the DataFrame into training and testing sets, trains models, makes predictions, and evaluates performance.
    Performance metrics are mean absolute error and r2 score, which are printed and visualized with line plots.

    Parameters:
    df (pandas.DataFrame): DataFrame with features and target.
    target_column (str): Target column name.
    test_size (float, optional): Proportion of data for test split. Default is 0.2.
    random_state (int, optional): Seed for random number generator in train-test split. Default is 42.

    Returns:
    None. Outputs performance metrics and line plots.
    """
    # Split the data into training and testing sets
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define the models to use
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }

    # Train each model and calculate the mean absolute error and r2 score
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append([model_name, mae, r2])

    # Convert the results to a DataFrame and visualize them
    results_df = pd.DataFrame(
        results, columns=["Model", "Mean Absolute Error", "R2 Score"]
    )
    print(results_df)

    plt.figure(figsize=(20, 5))
    sns.lineplot(y_test.values, label="Actual")
    sns.lineplot(predictions, label="Predicted")
    plt.title(f"{model_name} Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel(target_column)
    plt.legend()
    plt.show()


experiment_models(
    df, "Stage1.Output.Measurement0.U.Actual", test_size=0.2, random_state=42
)
