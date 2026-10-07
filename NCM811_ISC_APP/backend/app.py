from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static")
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "NCM811_final_ISC_model.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "isc_feature_columns.joblib"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

print("Model loaded")
print("Features:", len(feature_columns))


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(df):

    # ----------------------------------------------
    # Measurement block 1
    # ----------------------------------------------

    current1 = pd.to_numeric(
        df["电流/A"], errors="coerce"
    ).dropna()

    capacity1 = pd.to_numeric(
        df["容量/Ah"], errors="coerce"
    ).dropna()

    soc1 = pd.to_numeric(
        df["SOC|DOD/%"], errors="coerce"
    ).dropna()

    voltage1 = pd.to_numeric(
        df["电压/V"], errors="coerce"
    ).dropna()


    # ----------------------------------------------
    # Measurement block 2
    # ----------------------------------------------

    current2 = pd.to_numeric(
        df["电流/A.1"], errors="coerce"
    ).dropna()

    capacity2 = pd.to_numeric(
        df["容量/Ah.1"], errors="coerce"
    ).dropna()

    soc2 = pd.to_numeric(
        df["SOC|DOD/%.1"], errors="coerce"
    ).dropna()

    voltage2 = pd.to_numeric(
        df["电压/V.1"], errors="coerce"
    ).dropna()


    # ----------------------------------------------
    # Combine measurements
    # ----------------------------------------------

    current = pd.concat(
        [current1, current2],
        ignore_index=True
    )

    capacity = pd.concat(
        [capacity1, capacity2],
        ignore_index=True
    )

    soc_dod = pd.concat(
        [soc1, soc2],
        ignore_index=True
    )

    voltage = pd.concat(
        [voltage1, voltage2],
        ignore_index=True
    )


    # ----------------------------------------------
    # Derivatives
    # ----------------------------------------------

    d_voltage1 = voltage1.diff().dropna()
    d_voltage2 = voltage2.diff().dropna()

    d_current1 = current1.diff().dropna()
    d_current2 = current2.diff().dropna()

    d_voltage = pd.concat(
        [d_voltage1, d_voltage2],
        ignore_index=True
    )

    d_current = pd.concat(
        [d_current1, d_current2],
        ignore_index=True
    )


    # ----------------------------------------------
    # 28 FEATURES
    # ----------------------------------------------

    feature_data = {

        "Current_mean": current.mean(),
        "Current_std": current.std(),
        "Current_min": current.min(),
        "Current_max": current.max(),
        "Current_final": current.iloc[-1],

        "Capacity_mean": capacity.mean(),
        "Capacity_std": capacity.std(),
        "Capacity_min": capacity.min(),
        "Capacity_max": capacity.max(),
        "Capacity_final": capacity.iloc[-1],

        "SOC_DOD_mean": soc_dod.mean(),
        "SOC_DOD_std": soc_dod.std(),
        "SOC_DOD_min": soc_dod.min(),
        "SOC_DOD_max": soc_dod.max(),
        "SOC_DOD_final": soc_dod.iloc[-1],

        "Voltage_mean": voltage.mean(),
        "Voltage_std": voltage.std(),
        "Voltage_min": voltage.min(),
        "Voltage_max": voltage.max(),
        "Voltage_final": voltage.iloc[-1],

        "dVoltage_mean": d_voltage.mean(),
        "dVoltage_std": d_voltage.std(),
        "dVoltage_min": d_voltage.min(),
        "dVoltage_max": d_voltage.max(),

        "dCurrent_mean": d_current.mean(),
        "dCurrent_std": d_current.std(),
        "dCurrent_min": d_current.min(),
        "dCurrent_max": d_current.max()
    }


    features = pd.DataFrame([feature_data])

    # Exact training feature order
    features = features[feature_columns]

    return features


# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==================================================
# CSV PREDICTION
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No CSV file uploaded."
            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({
                "error": "No file selected."
            }), 400


        if not file.filename.lower().endswith(".csv"):

            return jsonify({
                "error": "Please upload a CSV file."
            }), 400


        df = pd.read_csv(file)

        features = extract_features(df)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]


        return jsonify({

            "result":
                "Internal Short Circuit (ISC)"
                if prediction == 1
                else "Normal",

            "prediction": int(prediction),

            "normal_probability":
                round(float(probabilities[0]) * 100, 2),

            "isc_probability":
                round(float(probabilities[1]) * 100, 2),

            "rows_processed": int(len(df)),

            "features_used": len(feature_columns),

            "input_type": "CSV"

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
