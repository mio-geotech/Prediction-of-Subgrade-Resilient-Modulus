"""
Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models

Script 01: Model Training, Evaluation, and Global SHAP Analysis

This script performs:

1. Data loading and preprocessing
2. Correlation analysis
3. Train-test splitting
4. Training and evaluation of four gradient boosting models
5. Export of model predictions and performance metrics
6. Global SHAP feature-importance analysis

Models included:
- LightGBM
- CatBoost
- XGBoost
- GBM

Execution environment:
This script was developed and executed in Google Colab. Google Colab-specific
commands for package installation, interactive file upload, and file download
have been intentionally retained to preserve the original research workflow.
"""


# ============================================================
# PACKAGE INSTALLATION
# ============================================================

!pip -q install xgboost lightgbm catboost shap openpyxl


# ============================================================
# LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from math import sqrt
from google.colab import files
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


shap.initjs()


# ============================================================
# UPLOAD THE EXCEL DATASET
# ============================================================

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

df_raw = pd.read_excel(file_name)

print("First five rows of the uploaded dataset:")

df_raw.head()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

df = df_raw.copy()

for column in df.columns:

    # Convert comma decimal separators to periods
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    # Convert values to numeric format
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


print("Dataset information:")

df.info()

df.head()


# ============================================================
# CORRELATION MATRIX
# ============================================================

feature_cols_original = [
    "No.4",
    "No.10",
    "No.40",
    "No.200",
    "LL",
    "PI",
    "wopt",
    "ρd,max",
    "CBRd",
    "CBRw",
    "Ec"
]

target_col = "MR"


plt.figure(figsize=(10, 8))

corr = df[
    feature_cols_original + [target_col]
].corr()


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)


plt.title(
    "Correlation Matrix of Input Variables and MR"
)

plt.tight_layout()

plt.show()


# ============================================================
# COLUMN NAME PREPARATION
# ============================================================

# Convert column names into formats compatible with LightGBM
df.columns = (
    df.columns
    .str.strip()
    .str.replace(
        "[^0-9a-zA-Z_]",
        "_",
        regex=True
    )
)


print("Modified column names:")

print(df.columns)


# ============================================================
# INPUT AND TARGET VARIABLES
# ============================================================

feature_cols = [
    "No_4",
    "No_10",
    "No_40",
    "No_200",
    "LL",
    "PI",
    "wopt",
    "_d_max",
    "CBRd",
    "CBRw",
    "Ec"
]


target_col = "MR"


X = df[feature_cols].copy()

y = df[target_col].copy()


print("X shape:", X.shape)

print("y shape:", y.shape)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)


# ============================================================
# GRADIENT BOOSTING MODELS
# ============================================================

models = {

    "GBM": GradientBoostingRegressor(
        random_state=42
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        n_estimators=500,
        loss_function="RMSE",
        random_state=42,
        verbose=0
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def mape(y_true, y_pred):
    """
    Calculate the mean absolute percentage error.

    Observations with zero experimental values are excluded
    to prevent division-by-zero errors.
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    nonzero_mask = (
        y_true != 0
    )

    if not np.any(
        nonzero_mask
    ):

        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[nonzero_mask]
                    - y_pred[nonzero_mask]
                )
                / y_true[nonzero_mask]
            )
        )
        * 100
    )

# ============================================================
# MODEL TRAINING, EVALUATION, AND EXPORT
# ============================================================

def evaluate_and_export(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    target_col
):

    # Train the model
    model.fit(
        X_train,
        y_train
    )


    # Generate predictions
    y_pred_train = model.predict(
        X_train
    )

    y_pred_test = model.predict(
        X_test
    )


    # Calculate mean squared errors
    mse_train = mean_squared_error(
        y_train,
        y_pred_train
    )

    mse_test = mean_squared_error(
        y_test,
        y_pred_test
    )


    # Store performance metrics
    metrics = {

        "model": model_name,

        "R2_train": r2_score(
            y_train,
            y_pred_train
        ),

        "R2_test": r2_score(
            y_test,
            y_pred_test
        ),

        "MSE_train": mse_train,

        "MSE_test": mse_test,

        "RMSE_train": sqrt(
            mse_train
        ),

        "RMSE_test": sqrt(
            mse_test
        ),

        "MAE_train": mean_absolute_error(
            y_train,
            y_pred_train
        ),

        "MAE_test": mean_absolute_error(
            y_test,
            y_pred_test
        ),

        "MAPE_train": mape(
            y_train,
            y_pred_train
        ),

        "MAPE_test": mape(
            y_test,
            y_pred_test
        )
    }


    print(
        f"\n=== {model_name} ==="
    )


    for metric_name, metric_value in metrics.items():

        if metric_name != "model":

            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )


    # Create the training prediction table
    train_df = X_train.copy()

    train_df[target_col] = y_train.values

    train_df[
        f"{target_col}_pred"
    ] = y_pred_train


    # Create the testing prediction table
    test_df = X_test.copy()

    test_df[target_col] = y_test.values

    test_df[
        f"{target_col}_pred"
    ] = y_pred_test


    # Define the output file name
    output_file = (
        f"{model_name}_predictions.xlsx"
    )


    # Export predictions and metrics
    with pd.ExcelWriter(
        output_file
    ) as writer:

        train_df.to_excel(
            writer,
            sheet_name="train",
            index=False
        )

        test_df.to_excel(
            writer,
            sheet_name="test",
            index=False
        )

        pd.DataFrame(
            [metrics]
        ).to_excel(
            writer,
            sheet_name="metrics",
            index=False
        )


    return (
        metrics,
        model,
        output_file
    )


# ============================================================
# RUN ALL MODELS
# ============================================================

all_metrics = []

fitted_models = {}

excel_files = []


for model_name, model in models.items():

    metrics, fitted_model, output_file = (
        evaluate_and_export(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            target_col
        )
    )


    all_metrics.append(
        metrics
    )

    fitted_models[
        model_name
    ] = fitted_model

    excel_files.append(
        output_file
    )


# ============================================================
# COMBINED PERFORMANCE TABLE
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)


print(
    "\nPerformance metrics of all models:"
)

display(
    metrics_df
)


metrics_df.to_excel(
    "Gradient_Boosting_Models_Metrics.xlsx",
    index=False
)


# ============================================================
# DOWNLOAD EXCEL OUTPUTS
# ============================================================

files.download(
    "Gradient_Boosting_Models_Metrics.xlsx"
)


files.download(
    "GBM_predictions.xlsx"
)

files.download(
    "LightGBM_predictions.xlsx"
)

files.download(
    "CatBoost_predictions.xlsx"
)

files.download(
    "XGBoost_predictions.xlsx"
)


# ============================================================
# GLOBAL SHAP ANALYSIS
# ============================================================

threshold = 0.80


X_train_sample = X_train.sample(
    min(
        100,
        len(X_train)
    ),
    random_state=42
)


for index, row in metrics_df.iterrows():

    model_name = row["model"]

    r2_test = row["R2_test"]


    # Skip models below the selected performance threshold
    if r2_test < threshold:

        continue


    model = fitted_models[
        model_name
    ]


    print(
        f"\n=== SHAP Analysis: "
        f"{model_name} "
        f"(Test R2 = {r2_test:.3f}) ==="
    )


    # TreeExplainer is used because all evaluated models
    # are tree-based gradient boosting algorithms
    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_train_sample
    )


    # Global SHAP feature-importance bar plot
    shap.summary_plot(
        shap_values,
        X_train_sample,
        feature_names=feature_cols,
        plot_type="bar",
        show=False
    )


    plt.title(
        f"{model_name} - "
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.show()


    # SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_train_sample,
        feature_names=feature_cols,
        show=False
    )


    plt.title(
        f"{model_name} - "
        "SHAP Summary Plot"
    )

    plt.tight_layout()

    plt.show()