"""
Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models

Script 02: SHAP Analysis and Dependence Plots

This script trains the reduced-feature Gradient Boosting Machine model
(GBM_KP), evaluates its predictive performance, calculates SHAP values,
generates SHAP dependence plots, and exports the underlying SHAP data
in Excel and CSV formats.

Execution environment:
This script was developed and executed in Google Colab. Google Colab-specific
commands (e.g., !pip install, files.upload(), and files.download()) are
intentionally retained to preserve the original research workflow.
"""


# ============================================================
# PACKAGE INSTALLATION
# ============================================================

!pip install -q shap openpyxl tifffile


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import io
import os
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from google.colab import files
from IPython.display import display

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

print("SHAP version:", shap.__version__)


# ============================================================
# UPLOAD AND READ THE DATASET
# ============================================================

uploaded = files.upload()

if not uploaded:
    raise RuntimeError("No input file was uploaded.")

file_name = next(iter(uploaded))

df = pd.read_excel(
    io.BytesIO(uploaded[file_name])
)

df.columns = [
    str(column).strip()
    for column in df.columns
]

print("Dataset:", file_name)
print("Dataset dimensions:", df.shape)
print("Columns:", df.columns.tolist())

display(df.head())


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

rename_map = {
    "PI": "PI",
    "CBRw": "CBRw",
    "CBR_w": "CBRw",
    "pd,max": "rho_d_max",
    "pd_max": "rho_d_max",
    "ρd,max": "rho_d_max",
    "ρd,max ": "rho_d_max",
    "rho_d_max": "rho_d_max",
    "wopt": "wopt",
    "w_opt": "wopt",
    "LL": "LL",
    "MR": "MR",
    "Mr": "MR",
    "M_R": "MR"
}

df = df.rename(
    columns={
        column: rename_map.get(column, column)
        for column in df.columns
    }
)


# ============================================================
# DEFINE INPUT AND TARGET VARIABLES
# ============================================================

features = [
    "PI",
    "CBRw",
    "rho_d_max",
    "wopt",
    "LL"
]

target = "MR"


missing_columns = [
    column
    for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# Convert model variables to numeric format
for column in features + [target]:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove observations containing missing values
model_df = (
    df[features + [target]]
    .dropna()
    .reset_index(drop=True)
)


# Assign a traceable identifier to each observation
model_df.insert(
    0,
    "Observation_ID",
    [
        f"OBS_{index:03d}"
        for index in range(1, len(model_df) + 1)
    ]
)

print(
    "Number of observations used:",
    len(model_df)
)

display(model_df.head())


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X = model_df[features].copy()

y = model_df[target].copy()

observation_ids = model_df[
    "Observation_ID"
].copy()


(
    X_train,
    X_test,
    y_train,
    y_test,
    id_train,
    id_test
) = train_test_split(
    X,
    y,
    observation_ids,
    test_size=0.20,
    random_state=42,
    shuffle=True
)


print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))


# ============================================================
# REDUCED-FEATURE GRADIENT BOOSTING MODEL
# ============================================================

# Default GradientBoostingRegressor parameters are retained.
# Only random_state is specified to ensure reproducibility.
gbm_kp = GradientBoostingRegressor(
    random_state=42
)


gbm_kp.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate the performance metrics used to evaluate the model.
    """

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "R2": r2_score(
            y_true,
            y_pred
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "MAPE_percent": (
            mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100
        )
    }


train_predictions = gbm_kp.predict(
    X_train
)

test_predictions = gbm_kp.predict(
    X_test
)


metrics_df = pd.DataFrame(
    [
        calculate_metrics(
            y_train,
            train_predictions
        ),
        calculate_metrics(
            y_test,
            test_predictions
        )
    ],
    index=[
        "Training",
        "Testing"
    ]
)


print("Model performance metrics:")

display(
    metrics_df.round(3)
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

# Combine training and testing data while preserving their indices
X_all = pd.concat(
    [
        X_train,
        X_test
    ],
    axis=0
)

y_all = pd.concat(
    [
        y_train,
        y_test
    ],
    axis=0
)

ids_all = pd.concat(
    [
        id_train,
        id_test
    ],
    axis=0
)


dataset_type = pd.Series(
    (
        ["Training"] * len(X_train)
        + ["Testing"] * len(X_test)
    ),
    index=X_all.index,
    name="Dataset_Type"
)


predictions_all = gbm_kp.predict(
    X_all
)


# Create a SHAP TreeExplainer
explainer = shap.TreeExplainer(
    gbm_kp
)

shap_values_array = explainer.shap_values(
    X_all
)


# Some SHAP versions may return a list
if isinstance(shap_values_array, list):
    shap_values_array = shap_values_array[0]


shap_values_array = np.asarray(
    shap_values_array
)


# Verify SHAP matrix dimensions
if shap_values_array.shape != X_all.shape:
    raise ValueError(
        f"Unexpected SHAP matrix dimensions: "
        f"{shap_values_array.shape}; "
        f"input matrix dimensions: {X_all.shape}"
    )


shap_df = pd.DataFrame(
    shap_values_array,
    columns=[
        f"SHAP_{feature}"
        for feature in features
    ],
    index=X_all.index
)


print(
    "SHAP matrix dimensions:",
    shap_df.shape
)

display(shap_df.head())


# ============================================================
# SHAP DEPENDENCE PLOTS
# ============================================================

feature_labels = {
    "PI": "Plasticity index, PI (%)",
    "CBRw": r"Soaked CBR, CBR$_w$ (%)",
    "rho_d_max": (
        r"Maximum dry unit density, "
        r"$\rho_{d,\max}$ (g/cm$^3$)"
    ),
    "wopt": (
        r"Optimum moisture content, "
        r"$w_{opt}$ (%)"
    ),
    "LL": "Liquid limit, LL (%)"
}


panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)"
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8.5)
)

axes = axes.flatten()


for index, feature in enumerate(features):

    axis = axes[index]

    feature_index = features.index(
        feature
    )


    shap.dependence_plot(
        ind=feature_index,
        shap_values=shap_values_array,
        features=X_all,
        feature_names=features,
        interaction_index="auto",
        ax=axis,
        show=False,
        dot_size=28,
        alpha=0.80
    )


    axis.axhline(
        0,
        linewidth=0.9,
        linestyle="--"
    )


    axis.set_title(
        (
            f"{panel_labels[index]} "
            f"{feature_labels[feature]}"
        ),
        fontsize=12,
        fontweight="bold",
        pad=10
    )


    axis.set_xlabel(
        feature_labels[feature],
        fontsize=11
    )


    axis.set_ylabel(
        (
            r"SHAP contribution to predicted "
            r"$M_R$ (kPa)"
        ),
        fontsize=11
    )


    axis.tick_params(
        axis="both",
        labelsize=10
    )


    axis.grid(
        linewidth=0.4,
        alpha=0.25
    )


# Disable the unused sixth panel
axes[-1].axis("off")


fig.tight_layout(
    pad=2.0,
    w_pad=2.5,
    h_pad=2.3
)


# Save the figure in publication-quality formats
plt.savefig(
    "GBM_KP_SHAP_Dependence.png",
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    "GBM_KP_SHAP_Dependence.tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)


plt.savefig(
    "GBM_KP_SHAP_Dependence.pdf",
    bbox_inches="tight"
)


plt.show()


# ============================================================
# CREATE THE WIDE-FORM SHAP DATA TABLE
# ============================================================

raw_wide_df = pd.DataFrame(
    {
        "Observation_ID": ids_all.values,
        "Dataset_Type": dataset_type.values,
        "Experimental_MR_kPa": y_all.values,
        "Predicted_MR_kPa": predictions_all,
        "Residual_Error_kPa": (
            predictions_all
            - y_all.values
        )
    },
    index=X_all.index
)


raw_wide_df = pd.concat(
    [
        raw_wide_df.reset_index(
            drop=True
        ),
        X_all.reset_index(
            drop=True
        ),
        shap_df.reset_index(
            drop=True
        )
    ],
    axis=1
)


display(raw_wide_df.head())


# ============================================================
# CREATE THE LONG-FORM SHAP DATA TABLE
# ============================================================

long_records = []


for row_position in range(
    len(X_all)
):

    for feature in features:

        long_records.append(
            {
                "Observation_ID": (
                    ids_all.iloc[
                        row_position
                    ]
                ),
                "Dataset_Type": (
                    dataset_type.iloc[
                        row_position
                    ]
                ),
                "Feature": feature,
                "Feature_Value": (
                    X_all.iloc[
                        row_position
                    ][feature]
                ),
                "SHAP_Value_kPa": (
                    shap_df.iloc[
                        row_position
                    ][f"SHAP_{feature}"]
                ),
                "Experimental_MR_kPa": (
                    y_all.iloc[
                        row_position
                    ]
                ),
                "Predicted_MR_kPa": (
                    predictions_all[
                        row_position
                    ]
                ),
                "Residual_Error_kPa": (
                    predictions_all[
                        row_position
                    ]
                    - y_all.iloc[
                        row_position
                    ]
                )
            }
        )


raw_long_df = pd.DataFrame(
    long_records
)


display(
    raw_long_df.head(10)
)


# ============================================================
# CREATE FEATURE-SPECIFIC SHAP TABLES
# ============================================================

feature_raw_tables = {}


for feature in features:

    feature_table = raw_wide_df[
        [
            "Observation_ID",
            "Dataset_Type",
            "Experimental_MR_kPa",
            "Predicted_MR_kPa",
            "Residual_Error_kPa",
            feature,
            f"SHAP_{feature}"
        ]
    ].copy()


    feature_table = feature_table.rename(
        columns={
            feature: "Feature_Value",
            f"SHAP_{feature}": (
                "SHAP_Value_kPa"
            )
        }
    )


    feature_table[
        "Feature"
    ] = feature


    feature_table = feature_table[
        [
            "Observation_ID",
            "Dataset_Type",
            "Feature",
            "Feature_Value",
            "SHAP_Value_kPa",
            "Experimental_MR_kPa",
            "Predicted_MR_kPa",
            "Residual_Error_kPa"
        ]
    ]


    feature_table = (
        feature_table
        .sort_values(
            "Feature_Value"
        )
        .reset_index(
            drop=True
        )
    )


    feature_raw_tables[
        feature
    ] = feature_table


display(
    feature_raw_tables[
        "PI"
    ].head()
)


# ============================================================
# EXPORT RESULTS TO EXCEL
# ============================================================

excel_output = (
    "GBM_KP_SHAP_Raw_Data.xlsx"
)


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    metrics_df.to_excel(
        writer,
        sheet_name="Model_Metrics"
    )


    raw_wide_df.to_excel(
        writer,
        sheet_name="All_Data_Wide",
        index=False
    )


    raw_long_df.to_excel(
        writer,
        sheet_name="All_Data_Long",
        index=False
    )


    for feature, table in (
        feature_raw_tables.items()
    ):

        safe_sheet_name = {
            "rho_d_max": "rho_d_max",
            "wopt": "wopt"
        }.get(
            feature,
            feature
        )


        table.to_excel(
            writer,
            sheet_name=(
                safe_sheet_name[:31]
            ),
            index=False
        )


print(
    "Excel file created:",
    excel_output
)


# ============================================================
# EXPORT RESULTS TO CSV
# ============================================================

os.makedirs(
    "SHAP_Raw_CSV",
    exist_ok=True
)


raw_wide_df.to_csv(
    "SHAP_Raw_CSV/All_Data_Wide.csv",
    index=False
)


raw_long_df.to_csv(
    "SHAP_Raw_CSV/All_Data_Long.csv",
    index=False
)


metrics_df.to_csv(
    "SHAP_Raw_CSV/Model_Metrics.csv"
)


for feature, table in (
    feature_raw_tables.items()
):

    table.to_csv(
        (
            f"SHAP_Raw_CSV/"
            f"{feature}_SHAP_Data.csv"
        ),
        index=False
    )


print("CSV files saved.")


# ============================================================
# CREATE ZIP ARCHIVE
# ============================================================

zip_name = (
    "GBM_KP_SHAP_Results.zip"
)


files_to_zip = [
    "GBM_KP_SHAP_Dependence.png",
    "GBM_KP_SHAP_Dependence.tiff",
    "GBM_KP_SHAP_Dependence.pdf",
    excel_output
]


with zipfile.ZipFile(
    zip_name,
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    for file_path in files_to_zip:

        if os.path.exists(
            file_path
        ):

            zip_file.write(
                file_path,
                arcname=os.path.basename(
                    file_path
                )
            )


    for root, _, filenames in os.walk(
        "SHAP_Raw_CSV"
    ):

        for filename in filenames:

            full_path = os.path.join(
                root,
                filename
            )


            zip_file.write(
                full_path,
                arcname=os.path.join(
                    "SHAP_Raw_CSV",
                    filename
                )
            )


print(
    "ZIP archive created:",
    zip_name
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

files.download(
    zip_name
)