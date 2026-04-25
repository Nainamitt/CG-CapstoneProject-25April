import joblib
import pandas as pd
from utils import load_model

model = load_model("models/demand_model.pkl")

# Load trained model
model = joblib.load("models/demand_model.pkl")

# Sample input
data = pd.DataFrame([{
    "Product_ID": 101,
    "day": 1,
    "month": 5,
    "year": 2026,
    "weekday": 4,
    "Region_East": 0,
    "Region_North": 1,
    "Region_South": 0
}])

# Predict
prediction = model.predict(data)

print("Predicted Demand:", round(prediction[0], 2))