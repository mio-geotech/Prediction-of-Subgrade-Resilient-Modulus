"""
Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models

Script 03: Five-Fold Cross-Validation

This script evaluates the predictive performance and stability of four
gradient boosting models using five-fold cross-validation. It calculates
fold-specific and summary performance metrics, generates publication-quality
figures, and exports the numerical results in Excel and CSV formats.

Models included:
- LightGBM
- CatBoost
- XGBoost
- GBM

Execution environment:
This script was developed and executed in Google Colab. Google Colab-specific
commands (e.g., !pip install, files.upload(), and files.download()) have
intentionally been retained to preserve the original research workflow.
"""


# ============================================================
# PACKAGE INSTALLATION
# ============================================================

!pip install -q xgboost lightgbm catboost openpyxl


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import io
import os
import json
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# Use the manuscript font settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11


# Reproducibility and cross-validation settings
RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# UPLOAD AND READ THE DATASET
# ============================================================

uploaded = files.upload()


if not uploaded:
    raise RuntimeError(
        "No Excel dataset was uploaded."
    )


file_name = next(
    iter(uploaded)
)


df_raw = pd.read_excel(
    io.BytesIO(
        uploaded[file_name]
    )
)


print(
    "Uploaded dataset:",
    file_name
)

print(
    "Raw dataset dimensions:",
    df_raw.shape
)

print(
    "Raw column names:",
    df_raw.columns.tolist()
)


display(
    df_raw.head()
)


# ============================================================
# DATA PREPARATION
# ============================================================

df = df_raw.copy()


# Remove leading and trailing spaces from column names
df.columns = [
    str(column).strip()
    for column in df.columns
]


# Standardize alternative column names
rename_map = {
    "No.4": "No_4",
    "No_4": "No_4",

    "No.10": "No_10",
    "No_10": "No_10",

    "No.40": "No_40",
    "No_40": "No_40",

    "No.200": "No_200",
    "No_200": "No_200",

    "LL": "LL",
    "PI": "PI",

    "wopt": "wopt",
    "w_opt": "wopt",

    "ρd,max": "rho_d_max",
    "ρd,max ": "rho_d_max",
    "pd,max": "rho_d_max",
    "pd_max": "rho_d_max",
    "_d_max": "rho_d_max",
    "rho_d_max": "rho_d_max",

    "CBRd": "CBRd",
    "CBR_d": "CBRd",

    "CBRw": "CBRw",
    "CBR_w": "CBRw",

    "Ec": "Ec",
    "E_c": "Ec",

    "MR": "MR",
    "Mr": "MR",
    "M_R": "MR"
}


df = df.rename(
    columns={
        column: rename_map.get(
            column,
            column
        )
        for column in df.columns
    }
)


# ============================================================
# DEFINE INPUT AND TARGET VARIABLES
# ============================================================

feature_cols = [
    "No_4",
    "No_10",
    "No_40",
    "No_200",
    "LL",
    "PI",
    "wopt",
    "rho_d_max",
    "CBRd",
    "CBRw",
    "Ec"
]


target_col = "MR"


required_cols = (
    feature_cols
    + [target_col]
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_cols = [
    column
    for column in required_cols
    if column not in df.columns
]


if missing_cols:
    raise ValueError(
        f"Missing columns: {missing_cols}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in required_cols:

    # Convert comma decimal separators to periods
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    # Convert values to numeric format
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

number_of_rows_before_removal = len(
    df
)


model_df = (
    df[required_cols]
    .dropna()
    .reset_index(drop=True)
)


removed_rows = (
    number_of_rows_before_removal
    - len(model_df)
)


# Assign a traceable identifier to each observation
model_df.insert(
    0,
    "Observation_ID",
    np.arange(
        1,
        len(model_df) + 1
    )
)


print(
    "Number of observations included in the analysis:",
    len(model_df)
)

print(
    "Number of rows removed because of missing values:",
    removed_rows
)


display(
    model_df.head()
)


display(
    model_df[
        required_cols
    ].describe().T
)


# ============================================================
# DEFINE GRADIENT BOOSTING MODELS
# ============================================================

models = {

    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        n_estimators=500,
        loss_function="RMSE",
        random_state=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "GBM": GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
}


print(
    "Models included in the five-fold cross-validation analysis:"
)


for model_name in models:
    print(
        "-",
        model_name
    )


# ============================================================
# PERFORMANCE METRIC FUNCTIONS
# ============================================================

def safe_mape(y_true, y_pred):
    """
    Calculate the mean absolute percentage error after excluding
    observations with zero experimental target values.

    The result is expressed as a percentage.
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


def calculate_metrics(y_true, y_pred):
    """
    Calculate the model performance metrics for a validation fold.
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

        "RMSE": np.sqrt(
            mse
        ),

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "MAPE_percent": safe_mape(
            y_true,
            y_pred
        )
    }


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

X = model_df[feature_cols].copy()

y = model_df[target_col].copy()

observation_ids = model_df[
    "Observation_ID"
].copy()


kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


fold_results = []

oof_predictions = []


for model_name, base_model in models.items():

    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)


    for fold_number, (
        train_idx,
        valid_idx
    ) in enumerate(
        kf.split(X),
        start=1
    ):


        X_train_fold = X.iloc[
            train_idx
        ]

        X_valid_fold = X.iloc[
            valid_idx
        ]


        y_train_fold = y.iloc[
            train_idx
        ]

        y_valid_fold = y.iloc[
            valid_idx
        ]


        # Create a fresh copy of the model for each fold
        model = clone(
            base_model
        )


        model.fit(
            X_train_fold,
            y_train_fold
        )


        y_pred_fold = model.predict(
            X_valid_fold
        )


        metrics = calculate_metrics(
            y_valid_fold,
            y_pred_fold
        )


        fold_record = {

            "Model": model_name,

            "Fold": fold_number,

            "Training_n": len(
                train_idx
            ),

            "Validation_n": len(
                valid_idx
            ),

            **metrics
        }


        fold_results.append(
            fold_record
        )


        # Store all out-of-fold predictions
        for local_position, original_index in enumerate(
            valid_idx
        ):

            oof_predictions.append({

                "Model": model_name,

                "Fold": fold_number,

                "Observation_ID": int(
                    observation_ids.iloc[
                        original_index
                    ]
                ),

                "Experimental_MR": float(
                    y_valid_fold.iloc[
                        local_position
                    ]
                ),

                "Predicted_MR": float(
                    y_pred_fold[
                        local_position
                    ]
                ),

                "Residual_Error": float(
                    y_pred_fold[
                        local_position
                    ]
                    -
                    y_valid_fold.iloc[
                        local_position
                    ]
                )

            })


        print(

            f"Fold {fold_number:02d} | "

            f"R² = {metrics['R2']:.4f} | "

            f"RMSE = {metrics['RMSE']:.3f} | "

            f"MAE = {metrics['MAE']:.3f} | "

            f"MAPE = {metrics['MAPE_percent']:.3f}%"

        )


fold_results_df = pd.DataFrame(
    fold_results
)

oof_predictions_df = pd.DataFrame(
    oof_predictions
)


print(
    "\nFive-fold cross-validation completed."
)

display(
    fold_results_df.head(15)
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

metric_columns = [

    "R2",

    "MSE",

    "RMSE",

    "MAE",

    "MAPE_percent"

]


summary_rows = []


for model_name in models.keys():

    model_fold_df = fold_results_df[
        fold_results_df["Model"]
        ==
        model_name
    ]


    summary_record = {

        "Model": model_name,

        "Number_of_Folds": len(
            model_fold_df
        )

    }


    for metric in metric_columns:

        summary_record[
            f"{metric}_Mean"
        ] = model_fold_df[
            metric
        ].mean()


        summary_record[
            f"{metric}_SD"
        ] = model_fold_df[
            metric
        ].std(
            ddof=1
        )


        summary_record[
            f"{metric}_Min"
        ] = model_fold_df[
            metric
        ].min()


        summary_record[
            f"{metric}_Max"
        ] = model_fold_df[
            metric
        ].max()


    summary_rows.append(
        summary_record
    )


summary_df = pd.DataFrame(
    summary_rows
)


print(
    "Summary statistics of the five-fold cross-validation:"
)

display(
    summary_df.round(3)
)


# ============================================================
# MANUSCRIPT SUMMARY TABLE
# ============================================================

compact_summary_df = pd.DataFrame({

    "Model":
        summary_df["Model"],


    "R2_Mean_SD":
        summary_df.apply(

            lambda row:
            f"{row['R2_Mean']:.3f} ± "
            f"{row['R2_SD']:.3f}",

            axis=1
        ),


    "RMSE_Mean_SD":
        summary_df.apply(

            lambda row:
            f"{row['RMSE_Mean']:.1f} ± "
            f"{row['RMSE_SD']:.1f}",

            axis=1
        ),


    "MAE_Mean_SD":
        summary_df.apply(

            lambda row:
            f"{row['MAE_Mean']:.1f} ± "
            f"{row['MAE_SD']:.1f}",

            axis=1
        ),


    "MAPE_Mean_SD_percent":
        summary_df.apply(

            lambda row:
            f"{row['MAPE_percent_Mean']:.2f} ± "
            f"{row['MAPE_percent_SD']:.2f}",

            axis=1
        ),


    "R2_Range":
        summary_df.apply(

            lambda row:
            f"{row['R2_Min']:.3f}–"
            f"{row['R2_Max']:.3f}",

            axis=1
        )

})


print(
    "Compact summary table prepared for the manuscript."
)

display(
    compact_summary_df
)


# ============================================================
# BOXPLOTS OF CROSS-VALIDATION PERFORMANCE
# ============================================================

plot_metrics = {
    "R2": "Coefficient of determination, $R^2$",
    "RMSE": "RMSE (kPa)",
    "MAE": "MAE (kPa)",
    "MAPE_percent": "MAPE (%)"
}


model_order = [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "GBM"
]


for metric, ylabel in plot_metrics.items():

    plot_data = [

        fold_results_df.loc[
            fold_results_df["Model"] == model_name,
            metric
        ].values

        for model_name in model_order
    ]


    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )


    ax.boxplot(
        plot_data,
        tick_labels=model_order,
        showmeans=True
    )


    # Plot every fold value
    for x_position, values in enumerate(
        plot_data,
        start=1
    ):

        x_jitter = np.linspace(
            -0.06,
            0.06,
            len(values)
        )


        ax.scatter(
            np.full(
                len(values),
                x_position
            )
            + x_jitter,
            values,
            s=25,
            zorder=3
        )


    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        linewidth=0.5,
        alpha=0.30
    )


    fig.tight_layout()


    output_base = (
        f"CV_{metric}_Boxplot"
    )


    fig.savefig(
        f"{output_base}.png",
        dpi=600,
        bbox_inches="tight"
    )


    fig.savefig(
        f"{output_base}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={
            "compression": "tiff_lzw"
        }
    )


    fig.savefig(
        f"{output_base}.pdf",
        bbox_inches="tight"
    )


    plt.show()
    plt.close(fig)


# ============================================================
# FOLD-WISE R2 PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


for model_name in model_order:

    model_data = (
        fold_results_df[
            fold_results_df["Model"]
            ==
            model_name
        ]
        .sort_values("Fold")
    )


    ax.plot(
        model_data["Fold"],
        model_data["R2"],
        marker="o",
        linewidth=1.5,
        label=model_name
    )


ax.set_xlabel(
    "Cross-validation fold"
)

ax.set_ylabel(
    "$R^2$"
)


ax.set_xticks(
    range(
        1,
        N_SPLITS + 1
    )
)


ax.grid(
    linewidth=0.5,
    alpha=0.30
)


ax.legend(
    frameon=True
)


fig.tight_layout()


fig.savefig(
    "CV_Foldwise_R2.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "CV_Foldwise_R2.tiff",
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={
        "compression": "tiff_lzw"
    }
)

fig.savefig(
    "CV_Foldwise_R2.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE MODEL SETTINGS
# ============================================================

model_settings_rows = []


for model_name, model in models.items():

    model_parameters = model.get_params()

    model_settings_rows.append({

        "Model": model_name,

        "Parameters_JSON": json.dumps(
            model_parameters,
            ensure_ascii=False,
            default=str
        )

    })


model_settings_df = pd.DataFrame(
    model_settings_rows
)


display(
    model_settings_df
)


# ============================================================
# EXPORT RESULTS TO EXCEL
# ============================================================

excel_output = (
    "Gradient_Boosting_5Fold_CV_Results.xlsx"
)


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:


    fold_results_df.to_excel(
        writer,
        sheet_name="All_Fold_Results",
        index=False
    )


    summary_df.to_excel(
        writer,
        sheet_name="Summary_Statistics",
        index=False
    )


    compact_summary_df.to_excel(
        writer,
        sheet_name="Manuscript_Summary",
        index=False
    )


    oof_predictions_df.to_excel(
        writer,
        sheet_name="OOF_Predictions",
        index=False
    )


    model_settings_df.to_excel(
        writer,
        sheet_name="Model_Settings",
        index=False
    )


    model_df.to_excel(
        writer,
        sheet_name="Data_Used",
        index=False
    )


    for model_name in model_order:

        model_fold_results = (
            fold_results_df[
                fold_results_df["Model"]
                ==
                model_name
            ]
        )


        model_fold_results.to_excel(
            writer,
            sheet_name=f"{model_name}_Folds"[:31],
            index=False
        )


print(
    "Excel file successfully created:",
    excel_output
)


# ============================================================
# EXPORT RESULTS TO CSV
# ============================================================

output_folder = "CV_Outputs"

os.makedirs(
    output_folder,
    exist_ok=True
)


fold_results_df.to_csv(
    f"{output_folder}/All_Fold_Results.csv",
    index=False
)


summary_df.to_csv(
    f"{output_folder}/Summary_Statistics.csv",
    index=False
)


compact_summary_df.to_csv(
    f"{output_folder}/Manuscript_Summary.csv",
    index=False
)


oof_predictions_df.to_csv(
    f"{output_folder}/OOF_Predictions.csv",
    index=False
)


model_settings_df.to_csv(
    f"{output_folder}/Model_Settings.csv",
    index=False
)


print(
    "CSV files successfully created."
)


# ============================================================
# CREATE ZIP ARCHIVE
# ============================================================

zip_output = (
    "Gradient_Boosting_5Fold_CV_Results.zip"
)


files_to_zip = [

    excel_output,

    "CV_R2_Boxplot.png",
    "CV_R2_Boxplot.tiff",
    "CV_R2_Boxplot.pdf",

    "CV_RMSE_Boxplot.png",
    "CV_RMSE_Boxplot.tiff",
    "CV_RMSE_Boxplot.pdf",

    "CV_MAE_Boxplot.png",
    "CV_MAE_Boxplot.tiff",
    "CV_MAE_Boxplot.pdf",

    "CV_MAPE_percent_Boxplot.png",
    "CV_MAPE_percent_Boxplot.tiff",
    "CV_MAPE_percent_Boxplot.pdf",

    "CV_Foldwise_R2.png",
    "CV_Foldwise_R2.tiff",
    "CV_Foldwise_R2.pdf"

]


with zipfile.ZipFile(
    zip_output,
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
        output_folder
    ):

        for filename in filenames:

            full_path = os.path.join(
                root,
                filename
            )

            zip_file.write(
                full_path,
                arcname=os.path.join(
                    output_folder,
                    filename
                )
            )


print(
    "ZIP archive successfully created:",
    zip_output
)


# ============================================================
# DOWNLOAD OUTPUT FILES
# ============================================================

files.download(
    zip_output
)