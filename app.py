from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from io import StringIO
from google import genai

app = Flask(__name__)

# ================= GEMINI CONFIG =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ================= LOAD ML MODEL =================
model = joblib.load("model/carbon_model.pkl")
enc_vehicle = joblib.load("model/vehicle_encoder.pkl")
enc_food = joblib.load("model/food_encoder.pkl")
enc_work = joblib.load("model/work_encoder.pkl")

# ================= CSV SAFE LOADER =================
def load_uploaded_csv(file):
    raw = file.read().replace(b"\x00", b"")
    text = raw.decode("latin-1", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")

@app.route("/")
def home():
    return render_template("index.html")

# ================= MANUAL PREDICT =================
@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        km = float(request.form["km"])
        electricity = float(request.form["electricity"])
    except ValueError:
        return render_template("index.html", error="Invalid numeric input")

    vehicle = request.form["vehicle"]
    food = request.form["food"]
    work = request.form["work"]

    # Simplified emission factors (kg CO2)
    transport = km * 0.12
    electricity_part = (electricity * 0.82) / 30
    food_part = {"veg":2.5, "mixed":3.0, "non_veg":3.5}[food]
    work_part = {"wfh":1.5, "hybrid":2.2, "office":3.0}[work]

    prediction = round(
        transport + electricity_part + food_part + work_part, 2
    )

    breakdown = {
        "transport": round(transport, 2),
        "electricity": round(electricity_part, 2),
        "food": food_part,
        "work": work_part
    }

    return render_template(
        "index.html",
        manual_result=prediction,
        monthly_co2=round(prediction * 30, 2),
        yearly_co2=round((prediction * 365) / 1000, 2),
        chart_data=breakdown
    )

# ================= CSV PREDICT =================
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    df = load_uploaded_csv(request.files["file"])
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    required_cols = [
        "vehicle_type", "daily_km",
        "electricity_units_month", "food_type", "work_mode"
    ]
    if not all(col in df.columns for col in required_cols):
        return render_template("index.html", csv_error="Invalid CSV format")

    df["vehicle_type"] = enc_vehicle.transform(df["vehicle_type"].astype(str))
    df["food_type"] = enc_food.transform(df["food_type"].astype(str))
    df["work_mode"] = enc_work.transform(df["work_mode"].astype(str))

    X = df[required_cols]
    df["predicted_co2"] = model.predict(X)

    return render_template(
        "index.html",
        csv_result=round(df["predicted_co2"].mean(), 2)
    )

# ================= AI INSIGHTS =================
@app.route("/api/ai_insight", methods=["POST"])
def ai_insight():
    if not client:
        return jsonify({"answer": "AI service not configured", "reduction": {}})

    data = request.json
    total = data["total"]
    breakdown = data["breakdown"]
    question = data["question"]

    prompt = f"""
You are a sustainability AI assistant.
User daily CO₂ footprint: {total} kg/day
Breakdown: {breakdown}
Question: {question}
Give practical suggestions with estimated reductions.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    reduction = {
        k: round(v * r, 2)
        for k, v, r in zip(
            breakdown.keys(),
            breakdown.values(),
            [0.30, 0.25, 0.20, 0.15]
        )
    }

    return jsonify({"answer": response.text, "reduction": reduction})

if __name__ == "__main__":
    app.run(debug=True)
