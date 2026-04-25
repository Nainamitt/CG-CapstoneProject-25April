import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Retail_Demand_Forecasting")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_csv, prepare_date_features, encode_region

df = load_csv("data/sales_data.csv")
df = prepare_date_features(df)
df = encode_region(df)

# Load Data
df = pd.read_csv("data/sales_data.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Extract Date Features
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year
df["weekday"] = df["Date"].dt.weekday

# Encode Region
df = pd.get_dummies(df, columns=["Region"])

# Features / Target
X = df.drop(["Sales", "Date"], axis=1)
y = df["Sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow
mlflow.set_experiment("Retail_Demand_Forecasting")

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.log_model(model, "model")

# Save Model
joblib.dump(model, "models/demand_model.pkl")

print("Model Saved Successfully")