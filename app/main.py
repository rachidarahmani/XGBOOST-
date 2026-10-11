import os
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Wine Quality API", version="1.0.0")


@lru_cache
def get_artifact():
    path = os.getenv("MODEL_PATH", "models/wine_model.joblib")
    return joblib.load(path)


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=0)
    volatile_acidity: float = Field(..., ge=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., ge=0)
    chlorides: float = Field(..., ge=0)
    free_sulfur_dioxide: float = Field(..., ge=0)
    total_sulfur_dioxide: float = Field(..., ge=0)
    density: float = Field(..., gt=0)
    ph: float = Field(..., gt=0, lt=14)
    sulphates: float = Field(..., ge=0)
    alcohol: float = Field(..., ge=0)
    wine_type: Literal["red", "white"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    art = get_artifact()
    return {
        "metrics": art.get("metrics", {}),
        "importances": art.get("importances", {}),
    }


@app.post("/predict")
def predict(wine: WineFeatures):
    art = get_artifact()
    row = wine.model_dump()
    row["type"] = 0 if row.pop("wine_type") == "red" else 1

    X = pd.DataFrame([row])[art["features"]]

    for c, lo in art["lower"].items():  # même écrêtage qu'à l'entraînement
        X[c] = X[c].clip(lo, art["upper"][c])

    proba = float(art["model"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": "bon" if pred == 1 else "mauvais",
        "probability_good": round(proba, 4),
    }
