from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

print("Loading LSTM model...")
model  = load_model("flood_lstm_model.h5")
scaler = joblib.load("scaler.pkl")
print("Model loaded successfully!")

FEATURES = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
    "river_discharge",
    "dewpoint_2m_max",
    "soil_moisture_0_to_7cm",
    "soil_moisture_28_to_100cm",
    "rainfall_3d",
    "rainfall_7d",
    "humidity",
    "is_monsoon",
]

SEQUENCE_LENGTH = 5  # ✅ 30 se 5 kiya

# ── Last 5 days API se fetch karo ──
def fetch_last_5_days(lat, lon):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=9)  # 9 days isliye ki rolling sum ke liye buffer chahiye
    end_str    = end_date.strftime("%Y-%m-%d")
    start_str  = start_date.strftime("%Y-%m-%d")

    # Daily weather
    w_res = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude"  : lat, "longitude" : lon,
            "start_date": start_str, "end_date": end_str,
            "daily"     : [
                "precipitation_sum",
                "et0_fao_evapotranspiration",
                "precipitation_hours",
                "dewpoint_2m_max",
            ],
            "timezone": "Asia/Kolkata"
        }, timeout=15
    )
    df = pd.DataFrame(w_res.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])

    # River discharge
    f_res = requests.get(
        "https://flood-api.open-meteo.com/v1/flood",
        params={
            "latitude"  : lat, "longitude": lon,
            "daily"     : ["river_discharge"],
            "start_date": start_str, "end_date": end_str,
        }, timeout=15
    )
    df_r = pd.DataFrame(f_res.json()["daily"])
    df_r["time"] = pd.to_datetime(df_r["time"])
    df = df.merge(df_r, on="time", how="left")

    # Hourly soil moisture → daily average
    s_res = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude"  : lat, "longitude": lon,
            "start_date": start_str, "end_date": end_str,
            "hourly"    : [
                "soil_moisture_0_to_7cm",
                "soil_moisture_28_to_100cm",
            ],
            "timezone": "Asia/Kolkata"
        }, timeout=15
    )
    df_s         = pd.DataFrame(s_res.json()["hourly"])
    df_s["time"] = pd.to_datetime(df_s["time"])
    df_s["date"] = df_s["time"].dt.date
    df_soil      = df_s.groupby("date").mean(numeric_only=True).reset_index()
    df_soil["time"] = pd.to_datetime(df_soil["date"])
    df_soil      = df_soil.drop(columns=["date"])
    df = df.merge(df_soil, on="time", how="left")

    # ── Engineer features (training ke same) ──
    df = df.sort_values("time").reset_index(drop=True)
    df["rainfall_3d"] = df["precipitation_sum"].rolling(3).sum()
    df["rainfall_7d"] = df["precipitation_sum"].rolling(7).sum()
    df["month"]       = df["time"].dt.month
    df["is_monsoon"]  = df["month"].apply(lambda m: 1 if 6 <= m <= 9 else 0)
    df["humidity"]    = (100 - 5 * (
        df["dewpoint_2m_max"] - df["dewpoint_2m_max"].rolling(3).mean()
    )).clip(30, 99)

    df = df.ffill().fillna(0)

    # ✅ Sirf last 5 rows
    df = df.tail(SEQUENCE_LENGTH).reset_index(drop=True)

    print(f"Sequence dates: {df['time'].dt.strftime('%d-%b-%Y').tolist()}")
    return df


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ User ka lat/lon lo — default Bhopal
        lat = float(data.get("user_lat", 23.2599))
        lon = float(data.get("user_lon", 77.4126))

        print(f"\nFetching last 5 days for ({lat}, {lon})...")

        # ✅ API se last 5 days fetch karo automatically
        df = fetch_last_5_days(lat, lon)

        # Array banao (5, 11)
        arr = df[FEATURES].values

        # Pad if less than 5 rows
        if len(arr) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(arr), len(FEATURES)))
            arr = np.vstack([pad, arr])

        # Scale + reshape → (1, 5, 11)
        arr_sc = scaler.transform(arr)
        X      = arr_sc.reshape(1, SEQUENCE_LENGTH, len(FEATURES))

        # Predict
        probs = model.predict(X, verbose=0)[0]
        cls   = int(np.argmax(probs))

        label_map = {0: "Low", 1: "Medium", 2: "High"}
        prob_keys = ["Low", "Medium", "High"]
        prob_dict = {prob_keys[i]: float(probs[i]) for i in range(len(probs))}

        last = df.iloc[-1]
        print(f"Prediction: {label_map[cls]} ({probs[cls]*100:.1f}%)")

        return jsonify({
            "success"       : True,
            "risk_class"    : cls,
            "risk_label"    : label_map.get(cls, "Unknown"),
            "confidence"    : float(probs[cls]),
            "probabilities" : prob_dict,
            # ✅ Sequence info — debug ke liye
            "sequence_used" : df["time"].dt.strftime("%d-%b-%Y").tolist(),
            # ✅ Live values — frontend tiles ke liye
            "live_values"   : {
                "precipitation_sum"     : round(float(last["precipitation_sum"]), 2),
                "river_discharge"       : round(float(last["river_discharge"]), 3),
                "rainfall_3d"           : round(float(last["rainfall_3d"]), 2),
                "rainfall_7d"           : round(float(last["rainfall_7d"]), 2),
                "humidity"              : round(float(last["humidity"]), 1),
                "soil_moisture_0_to_7cm": round(float(last["soil_moisture_0_to_7cm"]), 4),
                "is_monsoon"            : int(last["is_monsoon"]),
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "ok",
        "features": FEATURES,
        "sequence": SEQUENCE_LENGTH
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)