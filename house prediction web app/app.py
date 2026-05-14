"""
House Price Prediction System
Flask backend — loads Gradient Boosting model and serves predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = os.path.join("model", "Gradient Boosting houseprediction.pkl")

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}")

# Exact feature order used during training
FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus",
]

# Categorical encoding maps
YES_NO_MAP = {"yes": 1, "no": 0}
FURNISHING_MAP = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}


def preprocess(form) -> pd.DataFrame:
    """Extract, validate, and encode form data into a model-ready DataFrame."""
    data = {
        "area":             int(form["area"]),
        "bedrooms":         int(form["bedrooms"]),
        "bathrooms":        int(form["bathrooms"]),
        "stories":          int(form["stories"]),
        "mainroad":         YES_NO_MAP[form["mainroad"]],
        "guestroom":        YES_NO_MAP[form["guestroom"]],
        "basement":         YES_NO_MAP[form["basement"]],
        "hotwaterheating":  YES_NO_MAP[form["hotwaterheating"]],
        "airconditioning":  YES_NO_MAP[form["airconditioning"]],
        "parking":          int(form["parking"]),
        "prefarea":         YES_NO_MAP[form["prefarea"]],
        "furnishingstatus": FURNISHING_MAP[form["furnishingstatus"]],
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def format_price(price: float) -> str:
    """Format predicted price as PKR with comma separators."""
    return f"PKR {int(round(price)):,}"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        df = preprocess(request.form)
        prediction = model.predict(df)[0]
        formatted_price = format_price(prediction)

        # Build a human-readable summary to pass to the result page
        summary = {
            "Area (sq ft)":        request.form["area"],
            "Bedrooms":            request.form["bedrooms"],
            "Bathrooms":           request.form["bathrooms"],
            "Stories":             request.form["stories"],
            "Parking Spaces":      request.form["parking"],
            "Main Road":           request.form["mainroad"].capitalize(),
            "Guest Room":          request.form["guestroom"].capitalize(),
            "Basement":            request.form["basement"].capitalize(),
            "Hot Water Heating":   request.form["hotwaterheating"].capitalize(),
            "Air Conditioning":    request.form["airconditioning"].capitalize(),
            "Preferred Area":      request.form["prefarea"].capitalize(),
            "Furnishing Status":   request.form["furnishingstatus"].replace("-", " ").title(),
        }

        return render_template("result.html", price=formatted_price, summary=summary)

    except (KeyError, ValueError) as e:
        error = f"Invalid input: {e}. Please fill all fields correctly."
        return render_template("index.html", error=error)
    except Exception as e:
        error = f"Prediction failed: {e}"
        return render_template("index.html", error=error)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
