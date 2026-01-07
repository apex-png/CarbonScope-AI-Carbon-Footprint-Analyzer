import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ================= PATH CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "carbon_emissions_dataset_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

required = {
    "vehicle_type",
    "daily_km",
    "electricity_units_month",
    "food_type",
    "work_mode"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ================= CO2 CALCULATION =================
def calculate_co2(row):
    # Simplified domain-based emission factors (kg CO2)
    vehicle = {"bike":0.03, "car":0.21, "bus":0.08, "metro":0.05}
    food = {"veg":2.5, "mixed":3.0, "non_veg":3.5}
    work = {"wfh":1.5, "hybrid":2.2, "office":3.0}

    return (
        row["daily_km"] * vehicle[row["vehicle_type"]] +
        (row["electricity_units_month"] * 0.82) / 30 +
        food[row["food_type"]] +
        work[row["work_mode"]]
    )

# Target variable derived from domain logic
df["co2_kg_per_day"] = df.apply(calculate_co2, axis=1)

# ================= ENCODING =================
enc_v = LabelEncoder()
enc_f = LabelEncoder()
enc_w = LabelEncoder()

df["vehicle_type"] = enc_v.fit_transform(df["vehicle_type"])
df["food_type"] = enc_f.fit_transform(df["food_type"])
df["work_mode"] = enc_w.fit_transform(df["work_mode"])

X = df[
    ["vehicle_type", "daily_km", "electricity_units_month", "food_type", "work_mode"]
]
y = df["co2_kg_per_day"]

# ================= TRAIN / TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL TRAINING =================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ================= EVALUATION =================
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"📊 Model RMSE: {rmse:.3f} kg CO2/day")

# ================= SAVE ARTIFACTS =================
joblib.dump(model, os.path.join(MODEL_DIR, "carbon_model.pkl"))
joblib.dump(enc_v, os.path.join(MODEL_DIR, "vehicle_encoder.pkl"))
joblib.dump(enc_f, os.path.join(MODEL_DIR, "food_encoder.pkl"))
joblib.dump(enc_w, os.path.join(MODEL_DIR, "work_encoder.pkl"))

print("✅ Training complete")
