
"""
FastAPI header model service - G6+G5 (24 features)
Run: uvicorn header_service:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("header_model.joblib")
FEATURE_VERSION = "G6+G5_v2"

app = FastAPI(title="Header Analysis Model API", version="2.0")

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    feature_names = list(model_data["features"])
    print(f"Loaded model with {len(feature_names)} features ({FEATURE_VERSION})")
except Exception as exc:
    model_data = None
    model = None
    feature_names = []
    startup_error = str(exc)
    print(f"ERROR loading model: {startup_error}")
else:
    startup_error = ""


class EmailRequest(BaseModel):
    email_id: str
    headers: Optional[Dict[str, Any]] = None
    header_features: Dict[str, Any]
    body_text: Optional[str] = ""
    attachments: Optional[List[Dict[str, Any]]] = []
    body_urls: Optional[List[str]] = []


class HeaderPredictionResponse(BaseModel):
    email_id: str
    message_id: str
    timestamp: str
    prediction: str
    confidence: float
    malicious_probability: float
    legitimate_probability: float
    source_path: str
    feature_version: str
    feature_count: int
    used_features: List[str]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "running" if model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "feature_count": len(feature_names),
        "feature_version": FEATURE_VERSION,
        "startup_error": startup_error,
    }


@app.post("/predict-header", response_model=HeaderPredictionResponse)
async def predict_header(request: EmailRequest) -> HeaderPredictionResponse:
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model is not loaded: {startup_error}")

    try:
        features = request.header_features or {}
        input_dict = {col: features.get(col, 0) for col in feature_names}
        X = pd.DataFrame([input_dict], columns=feature_names)

        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)

        malicious_idx = None
        legitimate_idx = None
        for idx, cls in enumerate(classes):
            cls_text = str(cls).lower()
            if cls_text in {"1", "malicious", "phishing", "spam"}:
                malicious_idx = idx
            if cls_text in {"0", "legitimate", "legit", "safe", "benign", "ham"}:
                legitimate_idx = idx

        if malicious_idx is None:
            malicious_idx = int(np.argmax(proba))
        if legitimate_idx is None:
            legitimate_idx = 1 - malicious_idx if len(proba) == 2 else int(np.argmin(proba))

        malicious_prob = float(proba[malicious_idx])
        legitimate_prob = float(proba[legitimate_idx])

        if malicious_prob >= legitimate_prob:
            prediction = "malicious"
            confidence = malicious_prob
        else:
            prediction = "legitimate"
            confidence = legitimate_prob

        headers = request.headers or {}
        message_id = str(headers.get("message_id") or "")

        return HeaderPredictionResponse(
            email_id=request.email_id,
            message_id=message_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=prediction,
            confidence=round(float(confidence), 4),
            malicious_probability=round(malicious_prob, 4),
            legitimate_probability=round(legitimate_prob, 4),
            source_path=str(MODEL_PATH),
            feature_version=FEATURE_VERSION,
            feature_count=len(feature_names),
            used_features=feature_names,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Header prediction failed: {exc}")
'@ | Out-File -FilePath "header_service.py" -Encoding UTF8'