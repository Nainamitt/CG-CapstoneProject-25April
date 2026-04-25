from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Demand Forecast API")

model = joblib.load("models/demand_model.pkl")

@app.get("/")
def home():
    return {"message": "Retail Demand Forecast API Running"}

@app.post("/predict-demand")
def predict(data: dict):

    date = pd.to_datetime(data["Date"])

    region = data["Region"]

    row = {
        "Product_ID": int(data["Product_ID"]),
        "day": date.day,
        "month": date.month,
        "year": date.year,
        "weekday": date.weekday(),
        "Region_East": 1 if region == "East" else 0,
        "Region_North": 1 if region == "North" else 0,
        "Region_South": 1 if region == "South" else 0
    }

    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]

    return {
        "Predicted_Demand": round(float(prediction), 2)
    }