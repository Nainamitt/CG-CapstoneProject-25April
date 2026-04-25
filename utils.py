import pandas as pd
import joblib

def load_csv(path):
    return pd.read_csv(path)

def prepare_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["weekday"] = df["Date"].dt.weekday
    return df

def encode_region(df):
    return pd.get_dummies(df, columns=["Region"])

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)