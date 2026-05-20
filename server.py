from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder="frontend")
CORS(app)

MODEL_DIR = "models"

model = joblib.load(f"{MODEL_DIR}/extra_trees_credit_model.pkl")
encoders = {
    col: joblib.load(f"{MODEL_DIR}/{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        input_df = pd.DataFrame({
            "Age": [int(data["age"])],
            "Sex": [encoders["Sex"].transform([data["sex"]])[0]],
            "Job": [int(data["job"])],
            "Housing": [encoders["Housing"].transform([data["housing"]])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([data["saving_accounts"]])[0]],
            "Checking account": [encoders["Checking account"].transform([data["checking_account"]])[0]],
            "Credit amount": [int(data["credit_amount"])],
            "Duration": [int(data["duration"])],
        })

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0].tolist()

        return jsonify({
            "prediction": int(pred),
            "label": "GOOD" if pred == 1 else "BAD",
            "confidence": round(max(proba) * 100, 1),
            "prob_good": round(proba[1] * 100, 1),
            "prob_bad": round(proba[0] * 100, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
