# Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models

This repository contains the Python source codes accompanying the manuscript:

**“Prediction of Subgrade Resilient Modulus for Compacted Soils Using a Hybrid Dataset and Gradient Boosting Models”**

## Overview

The provided scripts implement the machine learning workflow used to estimate the resilient modulus (MR) of compacted subgrade soils using a hybrid dataset.

The evaluated gradient boosting algorithms are:

- Light Gradient Boosting Machine (LightGBM)
- Categorical Boosting (CatBoost)
- Extreme Gradient Boosting (XGBoost)
- Gradient Boosting Machine (GBM)

The repository also includes the procedures used for:

- training and testing data separation;
- model training and prediction;
- statistical performance evaluation;
- five-fold cross-validation;
- SHAP-based model interpretation.

## Repository contents

- `01_Model_Training_Evaluation_and_SHAP.py`  
Implements data preprocessing, training and testing of LightGBM, CatBoost, XGBoost, and GBM models, calculation of statistical performance metrics, export of predictions, and global SHAP bar and summary analyses.

- `02_SHAP_Dependence_Analysis.ipynb`  
Generates the SHAP dependence plots for the five most influential input variables presented in the manuscript.

- `03_Five_Fold_Cross_Validation.ipynb`  
Performs the five-fold cross-validation analysis and reports fold-specific and average model-performance metrics.

- `04_GBM_Reduced_Feature_Model.py`  
Implements the reduced-input GBM model developed using the five SHAP-selected predictor variables.

## Data availability

The hybrid dataset used in the study combines newly generated experimental data with laboratory data compiled from the **Flexible Pavement Design Guide** published by the General Directorate of Highways (KGM, 2008).

Although the guide itself is publicly available, the original laboratory database underlying the guide is not publicly distributed. The database was accessed with permission exclusively for scientific research purposes within the scope of the R&D project acknowledged in the manuscript.

Consequently, the complete hybrid dataset cannot be redistributed through this repository. The experimental results generated in the present study are fully reported in the manuscript.

## Model configuration

The scripts include the final hyperparameter configurations adopted for the evaluated gradient boosting algorithms. Parameters not explicitly specified were retained at the default settings of the corresponding Python libraries.

A fixed random seed (`random_state = 42`) was used to improve reproducibility.

## Software requirements

The analyses were conducted using Python and the following principal libraries:

- NumPy
- pandas
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Matplotlib

The required packages are listed in `requirements.txt`.

## Usage

1. Prepare an input spreadsheet containing the required predictor variables and the target MR values.
2. Update the input file path or upload section in the Python script.
3. Run the model-training script.
4. Run the SHAP-analysis script after the models have been fitted.

The expected predictor columns are:

- No.4
- No.10
- No.40
- No.200
- LL
- PI
- wopt
- rho_d_max
- CBRd
- CBRw
- Ec

The target variable is:

- MR

## Citation

Please cite the associated manuscript when using or adapting the source codes contained in this repository.

## Contact

For questions concerning the implementation or supporting information, please contact the corresponding author.
