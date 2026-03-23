from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from map_generator import generate_flood_map
import os

app = Flask(__name__, static_folder="static")
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

SEQUENCE_LENGTH = 5


# ── Fetch last 5 days from APIs ──────────────────────────────────────────────
def fetch_last_5_days(lat, lon):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=9)
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

    # Hourly soil → daily average
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

    # Engineer features
    df = df.sort_values("time").reset_index(drop=True)
    df["rainfall_3d"] = df["precipitation_sum"].rolling(3).sum()
    df["rainfall_7d"] = df["precipitation_sum"].rolling(7).sum()
    df["month"]       = df["time"].dt.month
    df["is_monsoon"]  = df["month"].apply(lambda m: 1 if 6 <= m <= 9 else 0)
    df["humidity"]    = (100 - 5 * (
        df["dewpoint_2m_max"] - df["dewpoint_2m_max"].rolling(3).mean()
    )).clip(30, 99)

    df = df.ffill().fillna(0)
    df = df.tail(SEQUENCE_LENGTH).reset_index(drop=True)

    print(f"Sequence: {df['time'].dt.strftime('%d-%b').tolist()}")
    return df


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "flood_warning_dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        lat = float(data.get("user_lat", 23.2599))
        lon = float(data.get("user_lon", 77.4126))

        print(f"\nFetching last 5 days for ({lat}, {lon})...")

        df = fetch_last_5_days(lat, lon)

        arr = df[FEATURES].values

        if len(arr) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(arr), len(FEATURES)))
            arr = np.vstack([pad, arr])

        arr_sc = scaler.transform(arr)
        X      = arr_sc.reshape(1, SEQUENCE_LENGTH, len(FEATURES))

        probs = model.predict(X, verbose=0)[0]
        cls   = int(np.argmax(probs))

        label_map = {0: "Low", 1: "Medium", 2: "High"}
        prob_keys = ["Low", "Medium", "High"]
        prob_dict = {prob_keys[i]: float(probs[i]) for i in range(len(probs))}

        risk_label = label_map[cls]
        last       = df.iloc[-1]

        print(f"Prediction: {risk_label} ({probs[cls]*100:.1f}%)")

        # ── Live values for map + frontend ──
        live_values = {
            "precipitation_sum"     : round(float(last["precipitation_sum"]), 2),
            "river_discharge"       : round(float(last["river_discharge"]), 3),
            "rainfall_3d"           : round(float(last["rainfall_3d"]), 2),
            "rainfall_7d"           : round(float(last["rainfall_7d"]), 2),
            "humidity"              : round(float(last["humidity"]), 1),
            "soil_moisture_0_to_7cm": round(float(last["soil_moisture_0_to_7cm"]), 4),
            "is_monsoon"            : int(last["is_monsoon"]),
        }

        # ── Generate flood map ──
        try:
            os.makedirs("static", exist_ok=True)
            generate_flood_map(
                risk_label  = risk_label,
                confidence  = float(probs[cls]),
                live_values = live_values,
                lat         = lat,
                lon         = lon,
                output_path = "static/flood_map.html"
            )
            map_url = "/static/flood_map.html"
            print(f"Map generated: {map_url}")
        except Exception as map_err:
            print(f"Map generation failed: {map_err}")
            map_url = None

        return jsonify({
            "success"       : True,
            "risk_class"    : cls,
            "risk_label"    : risk_label,
            "confidence"    : float(probs[cls]),
            "probabilities" : prob_dict,
            "sequence_used" : df["time"].dt.strftime("%d-%b-%Y").tolist(),
            "live_values"   : live_values,
            "map_url"       : map_url,
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "ok",
        "sequence": SEQUENCE_LENGTH,
        "features": len(FEATURES),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)