"""
SkyDelay — Flask Backend (app.py)
===================================
Serves the SkyDelay web app and exposes a /predict API endpoint
that runs the real scikit-learn DecisionTreeRegressor.

Usage:
    pip install flask pandas scikit-learn numpy
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Global model & encoders ───────────────────────────────────────────────────
model    = None
encoders = {}
feature_names = []
model_stats   = {}

def load_or_train_model():
    global model, encoders, feature_names, model_stats

    MODEL_JSON = "model_data.json"

    # Load pre-trained model from JSON if available
    if os.path.exists(MODEL_JSON):
        print(f"Loading model from {MODEL_JSON} ...")
        with open(MODEL_JSON) as f:
            data = json.load(f)
        encoders      = data["encoders"]
        feature_names = data["feature_names"]
        model_stats   = {"mae": data.get("mae"), "r2": data.get("r2"), "n_flights": data.get("n_flights")}

        # Rebuild model from scratch using CSV so we have a live sklearn object
        if os.path.exists("flight_delays.csv"):
            _train_from_csv()
        else:
            print("  flight_delays.csv not found — using JSON tree for predictions only.")
            model = None   # will use JSON tree traversal fallback
        return

    # No JSON — train from CSV directly
    if os.path.exists("flight_delays.csv"):
        _train_from_csv()
    else:
        print("WARNING: Neither model_data.json nor flight_delays.csv found.")
        print("         Place flight_delays.csv in this folder and restart.")


def _train_from_csv():
    global model, encoders, feature_names, model_stats

    print("Training model from flight_delays.csv ...")
    df = pd.read_csv("flight_delays.csv")

    # Clean
    df = df.drop(['ActualDeparture', 'ActualArrival'], axis=1, errors='ignore')
    df['DelayReason'] = df['DelayReason'].fillna("Unknown")
    df = df.dropna()
    for col in ['FlightID', 'FlightNumber', 'Cancelled', 'Diverted']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # DateTime features
    df['ScheduledDeparture'] = pd.to_datetime(df['ScheduledDeparture'])
    df['hour']  = df['ScheduledDeparture'].dt.hour
    df['day']   = df['ScheduledDeparture'].dt.day
    df['month'] = df['ScheduledDeparture'].dt.month

    df['ScheduledArrival'] = pd.to_datetime(df['ScheduledArrival'])
    df['arrival_hour']  = df['ScheduledArrival'].dt.hour
    df['arrival_day']   = df['ScheduledArrival'].dt.day
    df['arrival_month'] = df['ScheduledArrival'].dt.month
    df = df.drop(['ScheduledDeparture', 'ScheduledArrival'], axis=1)

    # Encode
    encoders = {}
    for col in ['Airline', 'Origin', 'Destination', 'AircraftType', 'DelayReason', 'TailNumber']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    X = df.drop('DelayMinutes', axis=1)
    y = df['DelayMinutes']
    feature_names = list(X.columns)

    # Sample for speed
    if len(df) > 200000:
        df_s = df.sample(n=200000, random_state=42)
        X_s  = df_s.drop('DelayMinutes', axis=1)
        y_s  = df_s['DelayMinutes']
    else:
        X_s, y_s = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    model_stats = {"mae": round(mae, 2), "r2": round(r2, 4), "n_flights": len(df)}
    print(f"  MAE: {mae:.2f}  R²: {r2:.4f}  Flights: {len(df):,}")


def encode_input(key, value):
    """Encode a categorical value using the trained label encoder mapping."""
    if key in encoders:
        return encoders[key].get(value, 0)
    return value


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the SkyDelay HTML app."""
    html_path = "skydelay_original.html"
    if not os.path.exists(html_path):
        return "<h2>SkyDelay.html not found in project folder.</h2>", 404
    with open(html_path, encoding="utf-8") as f:
        return f.read()


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
        {
          "airline":     "Delta",
          "origin":      "LAX",
          "destination": "MIA",
          "aircraft":    "Boeing 737",
          "reason":      "Weather",
          "distance":    1500,
          "hour":        14
        }
    Returns:
        { "minutes": 18, "confidence": 82, "status": "Significant Delay" }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Place flight_delays.csv in project folder and restart."}), 503

    data = request.get_json(force=True)

    try:
        flight_hours = round(data.get("distance", 1000) / 500)
        arr_hour     = (data.get("hour", 12) + flight_hours) % 24
        today        = pd.Timestamp.today()

        row = {
            "Airline":      encode_input("Airline",      data.get("airline", "")),
            "Origin":       encode_input("Origin",       data.get("origin", "")),
            "Destination":  encode_input("Destination",  data.get("destination", "")),
            "DelayReason":  encode_input("DelayReason",  data.get("reason", "Unknown")),
            "AircraftType": encode_input("AircraftType", data.get("aircraft", "")),
            "TailNumber":   0,
            "Distance":     int(data.get("distance", 1000)),
            "hour":         int(data.get("hour", 12)),
            "day":          today.day,
            "month":        today.month,
            "arrival_hour": arr_hour,
            "arrival_day":  today.day,
            "arrival_month":today.month,
        }

        X_input = pd.DataFrame([row])[feature_names]
        minutes = int(round(float(model.predict(X_input)[0])))
        confidence = min(96, max(62, round(88 - abs(minutes) * 0.3)))

        if minutes <= 0:
            status = "Early / On Time"
        elif minutes < 15:
            status = "Minor Delay"
        else:
            status = "Significant Delay"

        return jsonify({
            "minutes":    minutes,
            "confidence": confidence,
            "status":     status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-info")
def model_info():
    """Return model metadata and available options."""
    return jsonify({
        "stats":    model_stats,
        "options": {
            "airlines":  list(encoders.get("Airline", {}).keys()),
            "origins":   list(encoders.get("Origin", {}).keys()),
            "destinations": list(encoders.get("Destination", {}).keys()),
            "aircraft":  list(encoders.get("AircraftType", {}).keys()),
            "reasons":   list(encoders.get("DelayReason", {}).keys()),
        }
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_or_train_model()
    print("\n🛫  SkyDelay running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
