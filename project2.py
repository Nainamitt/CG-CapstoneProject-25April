"""
Predictive Maintenance API

- Uses Azure ML endpoint
- Fully Pydantic v2 compliant (no deprecated usage)
- Handles IoT sensor data
- Includes anomaly detection + logging
"""

import os
import requests
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# ==============================
# LOAD ENV VARIABLES
# ==============================
load_dotenv()

AZURE_ENDPOINT = os.getenv("DeployEndpoint")
AZURE_KEY = os.getenv("DeployKey")

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("ENDPOINT and KEY must be set in .env file")

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict machine failure using IoT sensor data",
    version="1.0"
)

# ==============================
# INPUT SCHEMA (NO DEPRECATION)
# ==============================
class SensorInput(BaseModel):
    Timestamp: str = Field(..., json_schema_extra={"example": "2024-01-01 00:00:00"})
    MachineID: str = Field(..., json_schema_extra={"example": "M01"})
    Temperature: float = Field(..., json_schema_extra={"example": 78})
    Vibration: float = Field(..., json_schema_extra={"example": 0.89})
    Pressure: float = Field(..., json_schema_extra={"example": 40})
    Humidity: float = Field(..., json_schema_extra={"example": 51})

    @field_validator("Timestamp")
    @classmethod
    def validate_timestamp(cls, v: str):
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Timestamp must be YYYY-MM-DD HH:MM:SS")
        return v

# ==============================
# AZURE CALL FUNCTION
# ==============================
def call_azure(payload: dict):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_KEY}"
    }

    try:
        response = requests.post(
            AZURE_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Azure connection failed")

    if response.status_code != 200:
        logger.error(f"Azure error: {response.text}")
        raise HTTPException(status_code=500, detail="Azure prediction failed")

    try:
        return response.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid Azure response")

# ==============================
# ANOMALY DETECTION (AIOPS)
# ==============================
def detect_anomaly(data: SensorInput) -> bool:
    if data.Temperature > 100:
        return True
    if data.Vibration > 1.5:
        return True
    if data.Pressure > 80:
        return True
    return False

# ==============================
# API ENDPOINT
# ==============================
@app.post("/predict-failure")
def predict_failure(input_data: SensorInput):
    start_time = datetime.utcnow()

    try:
        # Azure ML Designer format
        payload = {
            "Inputs": {
                "input1": [
                    {
                        "Timestamp": input_data.Timestamp,
                        "MachineID": input_data.MachineID,
                        "Temperature": input_data.Temperature,
                        "Vibration": input_data.Vibration,
                        "Pressure": input_data.Pressure,
                        "Humidity": input_data.Humidity
                    }
                ]
            }
        }

        result = call_azure(payload)

        # Handle Azure output safely
        prediction = result.get("Results", result)

        anomaly = detect_anomaly(input_data)

        latency = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Prediction result: {prediction}")

        return {
            "failure_probability": prediction,
            "anomaly_detected": anomaly,
            "response_time_sec": latency,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)