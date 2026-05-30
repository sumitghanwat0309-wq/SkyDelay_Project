"""
SkyDelay — Model Training Script
=================================
Trains a DecisionTreeRegressor on flight_delays.csv,
exports the tree + encoders to model_data.json,
which is embedded inside SkyDelay.html for in-browser prediction.

Usage:
    pip install pandas scikit-learn numpy
    python train_model.py

Output:
    model_data.json  — tree + encoders (used by SkyDelay.html)
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading flight_delays.csv ...")
df = pd.read_csv("flight_delays.csv")
print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

# ── 2. Clean (matches notebook logic) ────────────────────────────────────────
df = df.drop(['ActualDeparture', 'ActualArrival'], axis=1, errors='ignore')
df['DelayReason'] = df['DelayReason'].fillna("Unknown")
df = df.dropna()

# Drop non-feature columns
for col in ['FlightID', 'FlightNumber', 'Cancelled', 'Diverted']:
    if col in df.columns:
        df = df.drop(col, axis=1)

# ── 3. DateTime features ──────────────────────────────────────────────────────
df['ScheduledDeparture'] = pd.to_datetime(df['ScheduledDeparture'])
df['hour']  = df['ScheduledDeparture'].dt.hour
df['day']   = df['ScheduledDeparture'].dt.day
df['month'] = df['ScheduledDeparture'].dt.month

df['ScheduledArrival'] = pd.to_datetime(df['ScheduledArrival'])
df['arrival_hour']  = df['ScheduledArrival'].dt.hour
df['arrival_day']   = df['ScheduledArrival'].dt.day
df['arrival_month'] = df['ScheduledArrival'].dt.month

df = df.drop(['ScheduledDeparture', 'ScheduledArrival'], axis=1)

# ── 4. Label Encode ───────────────────────────────────────────────────────────
encoders = {}
for col in ['Airline', 'Origin', 'Destination', 'AircraftType', 'DelayReason', 'TailNumber']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

print("  Unique values:")
for col in ['Airline', 'Origin', 'Destination', 'AircraftType', 'DelayReason']:
    if col in encoders:
        print(f"    {col}: {list(encoders[col].keys())}")

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
X = df.drop('DelayMinutes', axis=1)
y = df['DelayMinutes']

# Sample up to 200k rows for speed
if len(df) > 200000:
    df_s = df.sample(n=200000, random_state=42)
    X_s, y_s = df_s.drop('DelayMinutes', axis=1), df_s['DelayMinutes']
else:
    X_s, y_s = X, y

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

# ── 6. Train Model ────────────────────────────────────────────────────────────
print("\nTraining DecisionTreeRegressor ...")
model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MAE : {mae:.2f} min")
print(f"  R²  : {r2:.4f}")

# ── 7. Export Tree to JSON ────────────────────────────────────────────────────
def tree_to_dict(tree, feature_names):
    t = tree.tree_
    def recurse(node):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            return {"leaf": float(round(t.value[node][0][0], 2))}
        return {
            "feature":   int(t.feature[node]),
            "fname":     feature_names[t.feature[node]],
            "threshold": float(round(t.threshold[node], 4)),
            "left":      recurse(t.children_left[node]),
            "right":     recurse(t.children_right[node])
        }
    return recurse(0)

tree_dict = tree_to_dict(model, list(X.columns))

output = {
    "encoders":      encoders,
    "feature_names": list(X.columns),
    "tree":          tree_dict,
    "mae":           round(mae, 2),
    "r2":            round(r2, 4),
    "n_flights":     len(df)
}

with open("model_data.json", "w") as f:
    json.dump(output, f, separators=(',', ':'))

tree_size = len(json.dumps(tree_dict)) // 1024
print(f"\n  Tree size : {tree_size} KB")
print(f"  Saved     : model_data.json")
print("\nDone! Now open SkyDelay.html in your browser.")
