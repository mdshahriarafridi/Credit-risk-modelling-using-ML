# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses a conda-managed environment located at `./venv`.

```powershell
# Activate environment
conda activate D:\credit-risk-modelling-using-ml\venv

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

## Common Commands

```powershell
# Run the Streamlit web application
streamlit run app.py

# Launch the training notebook
jupyter notebook model_training.ipynb
```

## Architecture

This is an end-to-end credit risk classification project with two main components:

**`model_training.ipynb`** — Full ML pipeline from raw data to serialized artifacts:
1. Loads `german_credit_data.csv` (1,000 records, 11 columns)
2. EDA: univariate/multivariate analysis, skewness, kurtosis, correlation heatmaps
3. Preprocessing: drops nulls and unnamed index column, label-encodes categoricals (Sex, Job, Housing, Saving accounts, Checking account, Risk)
4. Trains and tunes four classifiers via GridSearchCV: Decision Tree, Random Forest, Extra Trees, XGBoost — all with `class_weight='balanced'`
5. Serializes the winning Extra Trees model and five label encoders to `models/` using joblib

**`app.py`** — Streamlit inference app that loads `models/extra_trees_credit_model.pkl` and the five encoder `.pkl` files, accepts 8 input features, and outputs a Good/Bad risk classification.

**`models/` directory** holds the deployment artifacts:
- `extra_trees_credit_model.pkl`
- `Sex_encoder.pkl`, `Housing_encoder.pkl`, `Saving accounts_encoder.pkl`, `Checking account_encoder.pkl`, `target_encoder.pkl`

## Key Details

- The encoder filenames contain spaces (e.g., `Saving accounts_encoder.pkl`) — preserve exact names when referencing them.
- Feature order passed to the model must match training order: Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration.
- scikit-learn version used during training must match the deployment environment; model `.pkl` files are not cross-version compatible.
- No test suite exists in this repository.
