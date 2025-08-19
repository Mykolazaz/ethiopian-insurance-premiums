from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd

app = FastAPI(title="Insurance Premium Prediction API")

model = xgb.XGBRegressor()
model.load_model("model/insurance_model.json")

class InsuranceInput(BaseModel):
    SEX: int
    INSR_TYPE: str
    INSURED_VALUE: float
    INSR_COVER: str
    TYPE_VEHICLE: str
    PROD_YEAR: int
    VEHICLE_AGE: int
    WAS_CLAIM_PAID: int
    PREVIOUS_CLAIM_PAID: int
    PREVIOUS_POLICYHOLDERS: int

@app.get("/")
def root():
    return {"message": "Insurance Premium Prediction API is running!"}

@app.post("/predict")
def predict(data: InsuranceInput):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    df["INSURED_VALUE"] = np.log1p(df["INSURED_VALUE"] + 1)

    df = pd.get_dummies(
        df,
        columns=["INSR_TYPE", "INSR_COVER", "TYPE_VEHICLE"],
        drop_first=True
    )

    missing_cols = set(model.get_booster().feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[model.get_booster().feature_names]

    prediction = model.predict(df)[0]
    return {"predicted_log_premium": float(prediction)}
