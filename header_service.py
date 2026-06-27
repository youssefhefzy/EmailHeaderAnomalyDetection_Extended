# header_service_compact.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os

# Load model from the same folder as this API file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "header_model.joblib")

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    FEATURE_NAMES = list(model_data["features"])
    print(f"Loaded header model with {len(FEATURE_NAMES)} features")
except Exception as e:
    raise RuntimeError(f"Failed to load header model: {e}")

app = FastAPI(title="Email Header Detection API", version="1.1.0")


class HeaderEmailRequest(BaseModel):
    """
    The API may receive the full parsed email JSON, but the model uses ONLY
    header_features. body_text, body_urls, attachments, etc. are ignored.
    """
    email_id: str
    headers: Optional[Dict[str, Any]] = None
    header_features: Dict[str, Any]

    class Config:
        extra = "ignore"  # ignore body_text, body_urls, attachments, and any other sections


class HeaderPredictionResponse(BaseModel):
    email_id: str
    message_id: Optional[str]
    timestamp: str
    prediction: str
    confidence: float


def get_header_value(headers: Optional[Dict[str, Any]], *keys: str) -> Optional[str]:
    """Case-insensitive lookup for header fields."""
    if not headers:
        return None

    lowered = {str(k).lower().replace("-", "_"): v for k, v in headers.items()}
    for key in keys:
        normalized = key.lower().replace("-", "_")
        value = lowered.get(normalized)
        if value not in (None, ""):
            return str(value)
    return None


def email_date_to_utc_iso(date_value: Optional[str]) -> str:
    """
    Convert email header date like:
    'Sun, 21 Jun 2026 21:36:03 +0300'
    to UTC ISO timestamp like:
    '2026-06-21T18:36:03+00:00'
    """
    if not date_value:
        return datetime.now(timezone.utc).isoformat()

    try:
        dt = parsedate_to_datetime(date_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        # If the email date is malformed, still return a valid API timestamp
        return datetime.now(timezone.utc).isoformat()


def class_to_label(pred_class: Any) -> str:
    """Map model class to the API label names."""
    if pred_class in (1, "1", True, "malicious", "phishing", "spam"):
        return "malicious"
    return "legitimate"


@app.post("/predict-header", response_model=HeaderPredictionResponse)
async def predict_header(request: HeaderEmailRequest):
    try:
        # Use ONLY the feature columns the trained model expects.
        # Extra features from OCR/parser JSON are ignored.
        row = {name: request.header_features.get(name, 0) for name in FEATURE_NAMES}
        X = pd.DataFrame([row], columns=FEATURE_NAMES)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Predict class and confidence.
        proba = model.predict_proba(X)[0]
        best_index = int(np.argmax(proba))
        pred_class = model.classes_[best_index]
        confidence = float(proba[best_index])

        message_id = get_header_value(request.headers, "message_id", "Message-ID")
        raw_date = get_header_value(request.headers, "date", "Date")

        return HeaderPredictionResponse(
            email_id=request.email_id,
            message_id=message_id,
            timestamp=email_date_to_utc_iso(raw_date),
            prediction=class_to_label(pred_class),
            confidence=round(confidence, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Header prediction failed: {e}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "expected_features": FEATURE_NAMES,
    }
