"""FastAPI service for ML model inference with API key authentication.

Provides endpoints for cancellation risk prediction and model metadata.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from utils import (
    build_preprocessor,
    engineer_features,
    get_classifiers,
    get_param_grids,
    load_and_clean_data,
    load_model_pipeline,
    save_model_pipeline,
    train_and_evaluate,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

API_KEY = os.getenv("API_KEY", "change-me-in-production")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key from the request header.

    Args:
        api_key: The API key from the X-API-Key header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the X-API-Key header.",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictionInput(BaseModel):
    """Input schema for cancellation prediction."""

    Month: int = Field(6, ge=1, le=12, description="Order month (1-12)")
    Qty: int = Field(1, ge=1, le=100, description="Quantity ordered")
    Amount: float = Field(500.0, ge=1, le=100000, description="Order amount in INR")
    IsWeekend: bool = Field(False, description="Is the delivery on a weekend?")
    HasPromotion: bool = Field(False, description="Does the order have a promotion?")
    Fulfilment: str = Field("Amazon", pattern=r"^(Amazon|Merchant)$")
    ServiceLevel: str = Field("Standard", pattern=r"^(Standard|Expedited)$")
    B2B: bool = Field(False, description="Is this a B2B order?")
    Category: str = Field("Set", description="Product category")


class PredictionOutput(BaseModel):
    """Output schema for cancellation prediction."""

    cancellation_probability: float
    risk_level: str
    model_used: str
    features_used: List[str]


class ModelInfo(BaseModel):
    """Schema for model metadata."""

    model_type: str
    metrics: Dict[str, float]
    feature_count: int
    trained_on: str


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="E-commerce Cancellation Predictor API",
    description="ML-powered API for predicting order cancellation risk. "
    "Protect endpoints with an API key via the X-API-Key header.",
    version="1.0.0",
)


@app.on_event("startup")
def _load_model() -> None:
    """Load or train the model on startup and store in app.state."""
    MODEL_PATH = "cancellation_model.pkl"
    pipeline = load_model_pipeline(MODEL_PATH)

    if pipeline is not None:
        app.state.model = pipeline["model"]
        app.state.features = pipeline["features"]
        app.state.metrics = pipeline.get("metrics", {})
        app.state.model_type = type(pipeline["model"]).__name__
        app.state.trained_on = datetime.now().isoformat()
        return

    dfs = load_and_clean_data()
    fe = engineer_features(dfs["sales"])
    target = fe["Cancelled"]

    numeric_cols = [
        "Qty", "Amount", "Month", "IsWeekend", "PricePerUnit",
        "HasPromotion", "IsHighValue", "Revenue_7d_MA",
    ]
    cat_cols = ["Fulfilment", "ship-service-level", "B2B"]

    X = fe[numeric_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    valid = target.notna() & X.notna().all(axis=1)
    X = X[valid]
    y = target[valid].astype(int)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = get_classifiers()
    grids = get_param_grids()
    best_name = list(classifiers.keys())[0]
    result = train_and_evaluate(
        classifiers[best_name], X_train, y_train, X_test, y_test, grids.get(best_name)
    )

    save_model_pipeline(result["model"], build_preprocessor(), numeric_cols + cat_cols, MODEL_PATH)

    app.state.model = result["model"]
    app.state.features = numeric_cols + cat_cols
    app.state.metrics = {
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "roc_auc": result["roc_auc"],
    }
    app.state.model_type = type(result["model"]).__name__
    app.state.trained_on = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> Dict[str, str]:
    return {
        "service": "E-commerce Cancellation Predictor API",
        "docs": "/docs",
        "status": "running",
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
def predict(
    input_data: PredictionInput,
    api_key: str = Security(verify_api_key),
) -> PredictionOutput:
    """Predict cancellation risk for a given order.

    Requires an API key in the X-API-Key header.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    fe = engineer_features(load_and_clean_data()["sales"])
    avg_price = fe["PricePerUnit"].median()
    avg_ma = fe["Revenue_7d_MA"].median()

    inp = pd.DataFrame([{
        "Qty": input_data.Qty,
        "Amount": input_data.Amount,
        "Month": input_data.Month,
        "IsWeekend": 1 if input_data.IsWeekend else 0,
        "PricePerUnit": input_data.Amount / input_data.Qty if input_data.Qty > 0 else avg_price,
        "HasPromotion": 1 if input_data.HasPromotion else 0,
        "IsHighValue": 1 if input_data.Amount > fe["Amount"].quantile(0.75) else 0,
        "Revenue_7d_MA": avg_ma,
        "Fulfilment": input_data.Fulfilment,
        "ship-service-level": input_data.ServiceLevel,
        "B2B": "True" if input_data.B2B else "False",
    }])
    for c in ["Fulfilment", "ship-service-level", "B2B"]:
        inp[c] = inp[c].astype(str)

    prob = app.state.model.predict_proba(inp)[0, 1]

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Moderate"
    else:
        risk = "High"

    return PredictionOutput(
        cancellation_probability=round(prob, 4),
        risk_level=risk,
        model_used=app.state.model_type,
        features_used=app.state.features,
    )


@app.get(
    "/model-info",
    response_model=ModelInfo,
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
def model_info(api_key: str = Security(verify_api_key)) -> ModelInfo:
    """Return metadata about the currently loaded model.

    Requires an API key in the X-API-Key header.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return ModelInfo(
        model_type=app.state.model_type,
        metrics=app.state.metrics,
        feature_count=len(app.state.features),
        trained_on=app.state.trained_on,
    )
