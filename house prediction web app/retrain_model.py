"""
retrain_model.py
Retrains the Gradient Boosting model on Housing.csv using the current
numpy/scikit-learn environment and saves a compatible .pkl file.
Run once: python retrain_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# ── Load data ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "Housing.csv")
df = pd.read_csv(CSV_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# ── Encode categoricals ───────────────────────────────────────────────────────
yes_no_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["furnishingstatus"] = df["furnishingstatus"].map(
    {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
)

# ── Features & target ─────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus",
]

X = df[FEATURE_COLUMNS]
y = df["price"]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train model ───────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nModel Performance:")
print(f"  R²  : {r2_score(y_test, y_pred):.4f}")
print(f"  MAE : {mean_absolute_error(y_test, y_pred):,.0f} PKR")

# ── Save compatible model ─────────────────────────────────────────────────────
OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "Gradient Boosting houseprediction.pkl"
)
joblib.dump(model, OUT_PATH)
print(f"\nModel saved to: {OUT_PATH}")
print("Done — you can now run: python app.py")
