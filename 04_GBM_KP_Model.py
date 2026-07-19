"""
Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models

Script 04: GBM KP Model

This script develops a Gradient Boosting Machine model using the five
most influential parameters identified through SHAP analysis. The model
is evaluated using an 80% training and 20% testing split.
Training and testing predictions are exported to an Excel workbook.

Input variables:
    PI      : Plasticity index
    CBRw    : Soaked California bearing ratio
    _d_max  : Maximum dry unit weight/density
    wopt    : Optimum moisture content
    LL      : Liquid limit

Target variable:
    MR      : Resilient modulus

Execution environment:
This script was developed and executed in Google Colab. Google Colab-specific
commands (e.g., !pip install, files.upload(), and files.download()) are
intentionally retained to preserve the original research workflow.
"""


# ============================================================
# PACKAGE INSTALLATION
# ============================================================

!pip install -q xlsxwriter


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd

from google.colab import files
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# UPLOAD AND READ THE DATASET
# ============================================================

uploaded = files.upload()


if not uploaded:
    raise RuntimeError(
        "No Excel file was uploaded."
    )


file_name = list(
    uploaded.keys()
)[0]


df_raw = pd.read_excel(
    file_name
)


print(
    "Uploaded file:",
    file_name
)


print(
    "\nFirst five rows of the raw dataset:"
)


display(
    df_raw.head()
)


# ============================================================
# DATA PREPARATION
# ============================================================

df = df_raw.copy()


# Convert comma decimal separators to periods and
# convert numeric-looking values to numeric data types
for column in df.columns:

    converted_column = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    numeric_column = pd.to_numeric(
        converted_column,
        errors="coerce"
    )

    # Replace the original column only when at least one
    # numeric value can be identified
    if numeric_column.notna().any():

        df[column] = numeric_column

    else:

        df[column] = converted_column


# Standardize column names so that they can be safely used
# in the Python workflow
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(
        r"[^0-9a-zA-Z_]",
        "_",
        regex=True
    )
)


print(
    "\nStandardized column names:"
)


print(
    df.columns.tolist()
)


print(
    "\nFirst five rows of the prepared dataset:"
)


display(
    df.head()
)


# ============================================================
# DEFINE INPUT AND TARGET VARIABLES
# ============================================================

# Five most influential variables identified through SHAP analysis
top5_cols = [
    "PI",
    "CBRw",
    "_d_max",
    "wopt",
    "LL"
]


target_col = "MR"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_cols = (
    top5_cols
    + [target_col]
)


missing_cols = [
    column
    for column in required_cols
    if column not in df.columns
]


if missing_cols:

    raise ValueError(
        "The following required columns are missing from the dataset: "
        f"{missing_cols}\n"
        f"Available columns: {df.columns.tolist()}"
    )


print(
    "\nAll required columns were found."
)


# ============================================================
# PREPARE MODEL DATA
# ============================================================

model_df = (
    df[required_cols]
    .copy()
)


# Ensure that all model variables are numeric
for column in required_cols:

    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


number_of_rows_before_cleaning = len(
    model_df
)


# Remove observations containing missing or nonnumeric values
model_df = (
    model_df
    .dropna(
        subset=required_cols
    )
    .reset_index(
        drop=True
    )
)


number_of_removed_rows = (
    number_of_rows_before_cleaning
    - len(model_df)
)


if model_df.empty:

    raise ValueError(
        "No valid observations remained after data preparation."
    )


print(
    "\nNumber of observations used:",
    len(model_df)
)


print(
    "Number of removed observations:",
    number_of_removed_rows
)


X = model_df[
    top5_cols
].copy()


y = model_df[
    target_col
].copy()


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "\nTraining input shape:",
    X_train.shape
)


print(
    "Testing input shape:",
    X_test.shape
)


# ============================================================
# PERFORMANCE METRIC FUNCTION
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
# DEFINE AND TRAIN THE GBM MODEL
# ============================================================

# Default GradientBoostingRegressor parameters are retained.
# Only random_state is specified to ensure reproducibility.
gbm_top5 = GradientBoostingRegressor(
    random_state=42
)


gbm_top5.fit(
    X_train,
    y_train
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

y_pred_train = gbm_top5.predict(
    X_train
)


y_pred_test = gbm_top5.predict(
    X_test
)


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

mse_train = mean_squared_error(
    y_train,
    y_pred_train
)


mse_test = mean_squared_error(
    y_test,
    y_pred_test
)


metrics = {
    "Model": "GBM_KP",

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

    "RMSE_train": np.sqrt(
        mse_train
    ),

    "RMSE_test": np.sqrt(
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
    "\n============================================================"
)


print(
    "GBM_KP MODEL PERFORMANCE"
)


print(
    "============================================================"
)


for metric_name, metric_value in metrics.items():

    if metric_name != "Model":

        print(
            f"{metric_name}: {metric_value:.4f}"
        )


# ============================================================
# CREATE TRAINING PREDICTION TABLE
# ============================================================

train_df = X_train.copy()


train_df[
    target_col
] = y_train.values


train_df[
    f"{target_col}_pred"
] = y_pred_train


# ============================================================
# CREATE TESTING PREDICTION TABLE
# ============================================================

test_df = X_test.copy()


test_df[
    target_col
] = y_test.values


test_df[
    f"{target_col}_pred"
] = y_pred_test


# ============================================================
# CREATE METRICS TABLE
# ============================================================

metrics_df = pd.DataFrame(
    [metrics]
)


print(
    "\nModel performance table:"
)


display(
    metrics_df.round(4)
)


# ============================================================
# EXPORT RESULTS TO EXCEL
# ============================================================

output_file = "GBM_KP_predictions.xlsx"


with pd.ExcelWriter(
    output_file,
    engine="xlsxwriter"
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

    metrics_df.to_excel(
        writer,
        sheet_name="metrics",
        index=False
    )


print(
    "\nExcel file successfully created:",
    output_file
)


# ============================================================
# DOWNLOAD THE OUTPUT FILE
# ============================================================

files.download(
    output_file
)