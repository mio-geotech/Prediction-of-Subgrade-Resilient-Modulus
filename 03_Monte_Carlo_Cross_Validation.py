"""
Prediction of Subgrade Resilient Modulus for Compacted Soils
Using a Hybrid Dataset and Gradient Boosting Models

Script 03: Monte Carlo Cross-Validation

This script evaluates the predictive performance and stability of four
gradient boosting models using repeated random sampling (Monte Carlo
cross-validation).

Execution environment:
This script was developed and executed in Google Colab.
Google Colab-specific commands (e.g., !pip install, files.upload(),
and files.download()) have intentionally been retained to preserve
the original research workflow.

Models included:
- LightGBM
- CatBoost
- XGBoost
- Gradient Boosting Machine (GBM)

Validation strategy:
- 1000 independent repetitions
- 80% training / 20% testing in each repetition
- A different random seed is used only for generating each train-test split
- Model hyperparameters, including random_state = 42, remain fixed
- Performance is evaluated exclusively on the unseen testing subset
- Test-stage R2, RMSE, and MAE are reported

The final hyperparameter configurations reported in the manuscript
are retained throughout the repeated validation procedure.
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

from sklearn.model_selection import train_test_split
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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11


# ============================================================
# MONTE CARLO SETTINGS
# ============================================================

N_REPETITIONS = 1000

TEST_SIZE = 0.20

# Used only to generate different train-test partitions
BASE_SPLIT_RANDOM_STATE = 42

# Fixed for all machine learning models in every repetition
MODEL_RANDOM_STATE = 42


# ============================================================
# UPLOAD AND READ THE DATASET
# ============================================================

uploaded = files.upload()

if not uploaded:
    raise RuntimeError(
        "No Excel dataset was uploaded."
    )

file_name = next(iter(uploaded))

df_raw = pd.read_excel(
    io.BytesIO(uploaded[file_name])
)

print("Uploaded dataset:", file_name)
print("Raw dataset dimensions:", df_raw.shape)
print("Raw column names:", df_raw.columns.tolist())

display(df_raw.head())


# ============================================================
# DATA PREPARATION
# ============================================================

df = df_raw.copy()

df.columns = [
    str(column).strip()
    for column in df.columns
]


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
        column: rename_map.get(column, column)
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

required_cols = feature_cols + [target_col]


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

    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

number_of_rows_before_removal = len(df)

model_df = (
    df[required_cols]
    .dropna()
    .reset_index(drop=True)
)

removed_rows = (
    number_of_rows_before_removal
    - len(model_df)
)


model_df.insert(
    0,
    "Observation_ID",
    np.arange(
        1,
        len(model_df) + 1
    )
)


print(
    "\nNumber of observations included in the analysis:",
    len(model_df)
)

print(
    "Number of rows removed because of missing values:",
    removed_rows
)

display(model_df.head())


# ============================================================
# CHECK EXPECTED SAMPLE SIZE
# ============================================================

if len(model_df) != 117:

    print(
        "\nWARNING:"
        f" The analysis contains {len(model_df)} observations "
        "instead of the 117 observations reported in the manuscript."
    )

else:

    print(
        "\nDataset size check completed successfully: "
        "117 observations."
    )


# ============================================================
# INPUT AND TARGET MATRICES
# ============================================================

X = model_df[feature_cols].copy()

y = model_df[target_col].copy()

observation_ids = model_df[
    "Observation_ID"
].copy()


# ============================================================
# DEFINE FINAL MODEL CONFIGURATIONS
# ============================================================

def create_models():

    """
    Create new model instances using the final parameter
    configurations reported in the manuscript.

    IMPORTANT:
    random_state remains fixed at 42 for every model and
    every Monte Carlo repetition. Only the train-test
    partition changes between repetitions.
    """

    return {

        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=MODEL_RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        ),

        "CatBoost": CatBoostRegressor(
            n_estimators=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_state=MODEL_RANDOM_STATE,
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
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1
        ),

        "GBM": GradientBoostingRegressor(
            random_state=MODEL_RANDOM_STATE
        )
    }


# ============================================================
# PERFORMANCE METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {

        "R2": r2_score(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(mse),

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        )
    }


# ============================================================
# MONTE CARLO CROSS-VALIDATION
# ============================================================

all_results = []

all_predictions = []


print(
    "\nStarting Monte Carlo cross-validation..."
)

print(
    f"Number of repetitions: {N_REPETITIONS}"
)

print(
    f"Training proportion: {(1 - TEST_SIZE) * 100:.0f}%"
)

print(
    f"Testing proportion: {TEST_SIZE * 100:.0f}%"
)

print(
    f"Fixed model random_state: {MODEL_RANDOM_STATE}"
)


for repetition in range(
    1,
    N_REPETITIONS + 1
):

    # Different seed only for the train-test partition
    split_seed = (
        BASE_SPLIT_RANDOM_STATE
        + repetition
        - 1
    )


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

        test_size=TEST_SIZE,

        random_state=split_seed,

        shuffle=True
    )


    # Model random states remain fixed
    models = create_models()


    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )


        y_pred_test = model.predict(
            X_test
        )


        metrics = calculate_metrics(
            y_test,
            y_pred_test
        )


        all_results.append({

            "Model": model_name,

            "Repetition": repetition,

            "Split_Random_State": split_seed,

            "Model_Random_State": MODEL_RANDOM_STATE,

            "Training_n": len(X_train),

            "Testing_n": len(X_test),

            "R2": metrics["R2"],

            "RMSE": metrics["RMSE"],

            "MAE": metrics["MAE"]
        })


        for local_position in range(
            len(X_test)
        ):

            all_predictions.append({

                "Model": model_name,

                "Repetition": repetition,

                "Split_Random_State": split_seed,

                "Observation_ID": int(
                    id_test.iloc[local_position]
                ),

                "Experimental_MR": float(
                    y_test.iloc[local_position]
                ),

                "Predicted_MR": float(
                    y_pred_test[local_position]
                ),

                "Residual_Error": float(
                    y_pred_test[local_position]
                    -
                    y_test.iloc[local_position]
                )
            })


    if (
        repetition % 100 == 0
        or repetition == 1
        or repetition == N_REPETITIONS
    ):

        print(
            f"Completed repetition "
            f"{repetition}/{N_REPETITIONS}"
        )


# ============================================================
# CREATE RESULTS DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)


print(
    "\nMonte Carlo cross-validation completed."
)

print(
    "Total model evaluations:",
    len(results_df)
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_rows = []


for model_name in [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "GBM"
]:

    model_results = results_df[
        results_df["Model"]
        ==
        model_name
    ]


    summary_rows.append({

        "Model": model_name,

        "Number_of_Repetitions":
            len(model_results),

        "R2_Mean":
            model_results["R2"].mean(),

        "R2_SD":
            model_results["R2"].std(
                ddof=1
            ),

        "R2_Min":
            model_results["R2"].min(),

        "R2_5th_Percentile":
            model_results["R2"].quantile(
                0.05
            ),

        "R2_25th_Percentile":
            model_results["R2"].quantile(
                0.25
            ),

        "R2_Median":
            model_results["R2"].median(),

        "R2_75th_Percentile":
            model_results["R2"].quantile(
                0.75
            ),

        "R2_95th_Percentile":
            model_results["R2"].quantile(
                0.95
            ),

        "R2_Max":
            model_results["R2"].max(),

        "Negative_R2_Count":
            int(
                (
                    model_results["R2"]
                    < 0
                ).sum()
            ),

        "RMSE_Mean":
            model_results["RMSE"].mean(),

        "RMSE_SD":
            model_results["RMSE"].std(
                ddof=1
            ),

        "RMSE_Min":
            model_results["RMSE"].min(),

        "RMSE_Max":
            model_results["RMSE"].max(),

        "MAE_Mean":
            model_results["MAE"].mean(),

        "MAE_SD":
            model_results["MAE"].std(
                ddof=1
            ),

        "MAE_Min":
            model_results["MAE"].min(),

        "MAE_Max":
            model_results["MAE"].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)


print(
    "\nMonte Carlo summary statistics:"
)

display(
    summary_df.round(3)
)


# ============================================================
# MANUSCRIPT-STYLE SUMMARY TABLE
# ============================================================

manuscript_summary_df = pd.DataFrame({

    "Model":
        summary_df["Model"],

    "R2_Mean_SD":
        summary_df.apply(
            lambda row:
            f"{row['R2_Mean']:.3f} ± "
            f"{row['R2_SD']:.3f}",
            axis=1
        ),

    "RMSE_Mean_SD_kPa":
        summary_df.apply(
            lambda row:
            f"{row['RMSE_Mean']:.1f} ± "
            f"{row['RMSE_SD']:.1f}",
            axis=1
        ),

    "MAE_Mean_SD_kPa":
        summary_df.apply(
            lambda row:
            f"{row['MAE_Mean']:.1f} ± "
            f"{row['MAE_SD']:.1f}",
            axis=1
        )
})


print(
    "\nManuscript-style summary table:"
)

display(
    manuscript_summary_df
)


# ============================================================
# R2 DISTRIBUTION BOXPLOT
# ============================================================

model_order = [
    "LightGBM",
    "CatBoost",
    "XGBoost",
    "GBM"
]


r2_plot_data = [

    results_df.loc[
        results_df["Model"]
        ==
        model_name,
        "R2"
    ].values

    for model_name in model_order
]


fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)


ax.boxplot(
    r2_plot_data,
    tick_labels=model_order,
    showmeans=True
)


ax.axhline(
    0,
    linestyle="--",
    linewidth=0.9
)


ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Test-stage $R^2$"
)


ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.30
)


fig.tight_layout()


fig.savefig(
    "Monte_Carlo_R2_Boxplot.png",
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    "Monte_Carlo_R2_Boxplot.pdf",
    bbox_inches="tight"
)


plt.show()
plt.close(fig)


# ============================================================
# RMSE DISTRIBUTION BOXPLOT
# ============================================================

rmse_plot_data = [

    results_df.loc[
        results_df["Model"]
        ==
        model_name,
        "RMSE"
    ].values

    for model_name in model_order
]


fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)


ax.boxplot(
    rmse_plot_data,
    tick_labels=model_order,
    showmeans=True
)


ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Test-stage RMSE (kPa)"
)


ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.30
)


fig.tight_layout()


fig.savefig(
    "Monte_Carlo_RMSE_Boxplot.png",
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    "Monte_Carlo_RMSE_Boxplot.pdf",
    bbox_inches="tight"
)


plt.show()
plt.close(fig)


# ============================================================
# MAE DISTRIBUTION BOXPLOT
# ============================================================

mae_plot_data = [

    results_df.loc[
        results_df["Model"]
        ==
        model_name,
        "MAE"
    ].values

    for model_name in model_order
]


fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)


ax.boxplot(
    mae_plot_data,
    tick_labels=model_order,
    showmeans=True
)


ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Test-stage MAE (kPa)"
)


ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.30
)


fig.tight_layout()


fig.savefig(
    "Monte_Carlo_MAE_Boxplot.png",
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    "Monte_Carlo_MAE_Boxplot.pdf",
    bbox_inches="tight"
)


plt.show()
plt.close(fig)


# ============================================================
# MODEL SETTINGS
# ============================================================

reference_models = create_models()


model_settings_rows = []


for model_name, model in reference_models.items():

    model_settings_rows.append({

        "Model": model_name,

        "Parameters_JSON": json.dumps(
            model.get_params(),
            ensure_ascii=False,
            default=str
        )
    })


model_settings_df = pd.DataFrame(
    model_settings_rows
)


# ============================================================
# EXPORT RESULTS TO EXCEL
# ============================================================

excel_output = (
    "Monte_Carlo_Cross_Validation_1000_Results.xlsx"
)


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:


    results_df.to_excel(
        writer,
        sheet_name="All_Repetitions",
        index=False
    )


    summary_df.to_excel(
        writer,
        sheet_name="Summary_Statistics",
        index=False
    )


    manuscript_summary_df.to_excel(
        writer,
        sheet_name="Manuscript_Summary",
        index=False
    )


    predictions_df.to_excel(
        writer,
        sheet_name="Test_Predictions",
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


print(
    "\nExcel file successfully created:",
    excel_output
)


# ============================================================
# EXPORT RESULTS TO CSV
# ============================================================

output_folder = (
    "Monte_Carlo_1000_Outputs"
)


os.makedirs(
    output_folder,
    exist_ok=True
)


results_df.to_csv(
    f"{output_folder}/All_Repetitions.csv",
    index=False
)


summary_df.to_csv(
    f"{output_folder}/Summary_Statistics.csv",
    index=False
)


manuscript_summary_df.to_csv(
    f"{output_folder}/Manuscript_Summary.csv",
    index=False
)


predictions_df.to_csv(
    f"{output_folder}/Test_Predictions.csv",
    index=False
)


model_settings_df.to_csv(
    f"{output_folder}/Model_Settings.csv",
    index=False
)


# ============================================================
# CREATE ZIP ARCHIVE
# ============================================================

zip_output = (
    "Monte_Carlo_Cross_Validation_1000_Results.zip"
)


files_to_zip = [

    excel_output,

    "Monte_Carlo_R2_Boxplot.png",
    "Monte_Carlo_R2_Boxplot.pdf",

    "Monte_Carlo_RMSE_Boxplot.png",
    "Monte_Carlo_RMSE_Boxplot.pdf",

    "Monte_Carlo_MAE_Boxplot.png",
    "Monte_Carlo_MAE_Boxplot.pdf"
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
# DOWNLOAD OUTPUT FILE
# ============================================================

files.download(
    zip_output
)