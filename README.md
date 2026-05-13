# Credit Risk Modelling Using Machine Learning

A machine learning project for predicting whether a loan applicant is likely to have a **good** or **bad** credit risk profile. The repository includes exploratory model training in a Jupyter notebook, serialized preprocessing artifacts, a trained classifier, and a Streamlit web app for interactive predictions.

## Project Overview

Credit risk modelling helps financial institutions estimate the likelihood that a borrower may default or become a risky customer. This project uses the German Credit dataset and applies supervised machine learning to classify applicants into credit risk categories.

The current Streamlit app serves a trained **Extra Trees Classifier** using applicant details such as age, job type, housing status, savings account status, checking account status, credit amount, and loan duration.

## Features

- End-to-end credit risk classification workflow
- Data preprocessing with label encoding for categorical variables
- Model comparison using multiple machine learning algorithms
- Algorithms used in this project `Decision Tree` , `Random Forest`, `Extra Trees Classifier`, `XGBoost`
- Saved model and encoder artifacts with `joblib`
- Interactive Streamlit app for real-time prediction
- Simple user interface for entering applicant information

## Dataset

The project uses `german_credit_data.csv`, which contains 1,000 credit records.

Main columns include:

- `Age`
- `Sex`
- `Job`
- `Housing`
- `Saving accounts`
- `Checking account`
- `Credit amount`
- `Duration`
- `Purpose`
- `Risk`

The target column is `Risk`, with two classes:

- `good`
- `bad`


## Model Training

The training workflow is available in `model_training.ipynb`.

The notebook includes:

- Loading and exploring the dataset
- Handling categorical variables with `LabelEncoder`
- Encoding the target variable
- Splitting data into training and testing sets
- Training and tuning multiple classifiers
- Saving the selected model and encoders


## Application Inputs

The Streamlit app expects the same feature names used during model training:

| Feature | Description |
|---|---|
| `Age` | Applicant age |
| `Sex` | Applicant gender |
| `Job` | Job category from 0 to 3 |
| `Housing` | Housing status: own, rent, or free |
| `Saving accounts` | Savings account level |
| `Checking account` | Checking account level |
| `Credit amount` | Requested credit amount |
| `Duration` | Loan duration in months |

Maintaining the exact feature names is important because scikit-learn validates prediction input columns against the columns used during training.

## Project Structure

```text
credit-risk-modelling-using-ml/
|-- app.py
|-- german_credit_data.csv
|-- model_training.ipynb
|-- models/
|   |-- extra_trees_credit_model.pkl
|   |-- Sex_encoder.pkl
|   |-- Housing_encoder.pkl
|   |-- Saving accounts_encoder.pkl
|   |-- Checking account_encoder.pkl
|   `-- target_encoder.pkl
|-- requirements.txt
`-- README.md
```

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd credit-risk-modelling-using-ml
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Start the app with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

Enter the applicant information and click **Predict Risk** to get the predicted credit risk category.

## Saved Artifacts

The app depends on the following saved files:

| File | Purpose |
|---|---|
| `models/extra_trees_credit_model.pkl` | Trained Extra Trees model used for prediction |
| `models/Sex_encoder.pkl` | Encoder for the `Sex` feature |
| `models/Housing_encoder.pkl` | Encoder for the `Housing` feature |
| `models/Saving accounts_encoder.pkl` | Encoder for the `Saving accounts` feature |
| `models/Checking account_encoder.pkl` | Encoder for the `Checking account` feature |
| `models/target_encoder.pkl` | Encoder for the target labels |

Do not rename these files unless you also update the loading logic in `app.py`.

## Important Notes

- The model input column names must match the training feature names exactly.
- `Credit amount` and `Checking account` are case-sensitive feature names.
- Missing values in the original dataset are represented in some categorical columns and should be handled consistently during retraining.
- If you retrain the model, save the updated model and encoders before running the app again.
- If scikit-learn shows a version warning while loading `.pkl` files, use the same scikit-learn version that was used when the model was trained.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Future Improvements

- Deploy the higher-performing XGBoost model if it generalizes well
- Add precision, recall, F1-score, and confusion matrix reporting
- Add probability scores alongside the final prediction
- Improve handling of missing categorical values
- Build a preprocessing pipeline to reduce manual encoder management
- Add model explainability using feature importance or SHAP values

## Disclaimer

This project is intended for educational and portfolio purposes. Credit decisions in real-world financial systems require stronger validation, fairness testing, regulatory review, monitoring, and human oversight.
