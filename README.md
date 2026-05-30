# ✈️ SkyDelay — Flight Delay Predictor

A machine learning web app that predicts flight delays using a real
DecisionTreeRegressor trained on 1,747,627 flight records.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `SkyDelay.html` | Frontend UI (works standalone OR served via Flask) |
| `app.py` | Flask backend — serves the app & exposes `/predict` API |
| `train_model.py` | Retrain the model on new CSV data |
| `model_data.json` | Exported decision tree + label encoders |
| `requirements.txt` | Python dependencies |
| `notebook_original.ipynb` | Original Jupyter notebook |
| `skydelay_original.html` | Original HTML (before ML integration) |
| `flight_delays.csv` | *(add your CSV here to retrain)* |

---

## 🚀 Quick Start

### Option A — Open directly (no install)
```
Open SkyDelay.html in any browser
```
The ML model is embedded in the HTML. Works fully offline.

### Option B — Run with Flask backend
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in browser
http://localhost:5000
```

---

## 🔗 How Everything Is Connected

```
flight_delays.csv
      │
      ▼
train_model.py ──► model_data.json
      │                   │
      │         ┌─────────┘
      ▼         ▼
    app.py  ◄── model_data.json
      │    (loads encoders + retrains sklearn model)
      │
      ├── GET  /              → serves SkyDelay.html
      ├── POST /predict       → returns { minutes, confidence, status }
      └── GET  /model-info    → returns available airlines, airports, etc.

SkyDelay.html
  (also works standalone — ML tree embedded as JS)
```

---

## 📡 API Reference

### `POST /predict`
```json
Request:
{
  "airline":     "Delta",
  "origin":      "LAX",
  "destination": "MIA",
  "aircraft":    "Boeing 737",
  "reason":      "Weather",
  "distance":    1500,
  "hour":        14
}

Response:
{
  "minutes":    22,
  "confidence": 82,
  "status":     "Significant Delay"
}
```

### `GET /model-info`
Returns model stats (MAE, R²) and all valid input options.

---

## 🔁 Retrain on New Data

```bash
# Place new CSV as flight_delays.csv, then:
python train_model.py
# Outputs updated model_data.json
# Restart app.py to use the new model
```

---

## 🧠 Model Details

- **Algorithm:** DecisionTreeRegressor (scikit-learn)
- **Training data:** 1,747,627 flights
- **MAE:** ~6.2 min  |  **R²:** 0.59
- **Features:** Airline, Origin, Destination, AircraftType, DelayReason, Distance, Hour, Day, Month

© 2026 SkyDelay
