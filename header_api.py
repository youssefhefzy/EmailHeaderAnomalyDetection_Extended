import app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timezone

# ─── Load the trained model at startup ──────────────────────
MODEL_PATH = "header_model.joblib"   # adjust path as needed
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
feature_names = model_data['features']   # list of feature names in correct order

# ─── Request body definition ────────────────────────────────
class EmailRequest(BaseModel):
    email_id: str
    headers: Optional[Dict[str, Any]] = None           # raw headers (if needed)
    header_features: Dict[str, Any]                     # the precomputed features
    body_text: Optional[str] = ""
    attachments: Optional[List[Dict]] = []
    body_urls: Optional[List[str]] = []

# ─── Response model ────────────────────────────────────────
class HeaderPredictionResponse(BaseModel):
    email_id: str
    timestamp: str
    prediction: str                # "legitimate" or "malicious"
    confidence: float              # probability of the predicted class
    source_path: str               # could be from request or server location
    feature_version: str           # optional, for traceability

# ─── Endpoint ──────────────────────────────────────────────
@app.post("/predict-header", response_model=HeaderPredictionResponse)
async def predict_header(request: EmailRequest):
    try:
        # 1. Build DataFrame from header_features, keeping only needed columns
        features = request.header_features
        # Select and order the columns according to the model's training
        input_dict = {col: features.get(col, 0) for col in feature_names}  # fill missing with 0 (or appropriate default)
        X = pd.DataFrame([input_dict])

        # 2. Predict
        proba = model.predict_proba(X)[0]   # shape (2,): [prob_legit, prob_malicious]
        pred_class = model.classes_[np.argmax(proba)]
        confidence = np.max(proba)

        # 3. Build response
        label = "legitimate" if pred_class == 0 else "malicious"   # adjust if your labels are 0=ham,1=spam
        return HeaderPredictionResponse(
            email_id=request.email_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=label,
            confidence=round(float(confidence), 4),
            source_path=f"header_model_{MODEL_PATH}",
            feature_version="G6_top30"   # adapt to the group used
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Header prediction failed: {str(e)}")