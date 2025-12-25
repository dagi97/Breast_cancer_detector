from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

dt_pipeline = joblib.load("models/dt_model.pkl")
lr_pipeline = joblib.load("models/lr_model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Features(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float

@app.post("/predict")
def predict(features: Features):
    input_data = pd.DataFrame([features.dict()])
    dt_pred = dt_pipeline.predict(input_data)[0]
    lr_pred = lr_pipeline.predict(input_data)[0]
    label_map = {0: "Benign", 1: "Malignant"}
    return {
        "decision_tree_prediction": label_map[dt_pred],
        "logistic_regression_prediction": label_map[lr_pred]
    }
