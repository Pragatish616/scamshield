# ================================================================
#  ScamShield — FastAPI Backend
#  Model: XGBoost + TF-IDF  |  Feature: scam_score risk metric
#
#  Endpoints:
#    GET  /              → health check
#    POST /predict       → single message prediction
#    POST /predict/batch → multiple messages at once
#    GET  /model/info    → model metadata
#
#  Render deployment:
#    Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
# ================================================================

import os
import math
import pickle
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scamshield")

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="ScamShield — AI Scam Detection API",
    description=(
        "Detects SMS scam/spam messages using XGBoost + TF-IDF. "
        "Returns prediction, probabilities, scam_score (0-100), "
        "risk level, and recommended action."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
#  MODEL LOADING
#  Searches multiple paths so it works both locally and on Render.
# ================================================================
def find_file(filename: str) -> str:
    """Search common paths for a file, return the first match."""
    search_paths = [
        f"models/{filename}",
        f"scamshield/{filename}",
        filename,
        f"/opt/render/project/src/models/{filename}",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{filename}'. "
        f"Searched: {search_paths}"
    )


def load_artifacts():
    """Load XGBoost model + TF-IDF vectorizer from disk."""
    model_path  = find_file("xgb_spam_model.pkl")
    tfidf_path  = find_file("tfidf_vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info(f"Model loaded from   : {model_path}")
    logger.info(f"Vectorizer loaded from: {tfidf_path}")
    return model, vectorizer


# Load once at startup — reused across all requests
try:
    MODEL, VECTORIZER = load_artifacts()
    logger.info("ScamShield model ready.")
except FileNotFoundError as e:
    logger.error(str(e))
    MODEL, VECTORIZER = None, None


# ================================================================
#  HELPER FUNCTIONS
# ================================================================
def get_scam_score(model, vec) -> float:
    """
    scam_score = spam_probability × 100  (range: 0 – 100)

    Works for any model:
      - predict_proba  → XGBoost, Random Forest, Logistic Regression
      - decision_function → SVM (sigmoid-transformed)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        return round(float(probs[1]) * 100, 2)

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        prob  = 1 / (1 + math.exp(-score))   # sigmoid
        return round(prob * 100, 2)

    # Hard fallback
    return 100.0 if int(model.predict(vec)[0]) == 1 else 0.0


def get_risk_level(scam_score: float) -> str:
    """
    Tiered risk classification based on scam_score.
      0  – 20  → Safe
      21 – 49  → Low
      50 – 79  → Medium
      80 – 100 → High
    """
    if scam_score >= 80:
        return "High"
    elif scam_score >= 50:
        return "Medium"
    elif scam_score >= 21:
        return "Low"
    return "Safe"


def get_action(label: str, risk_level: str) -> str:
    """Return human-readable recommended action."""
    actions = {
        ("spam", "High"): (
            "DANGER: Do NOT click any links or call any numbers in this message. "
            "Block the sender immediately. Report to cybercrime.gov.in or call 1930."
        ),
        ("spam", "Medium"): (
            "Suspicious message detected. Avoid sharing personal or banking details. "
            "Verify through the official website or app before taking any action."
        ),
        ("spam", "Low"): (
            "Possibly promotional or borderline content. "
            "Do not click unfamiliar links. When in doubt, ignore or delete."
        ),
    }
    if label == "ham":
        return (
            "Message appears legitimate. "
            "Still avoid clicking unknown links and never share OTPs with anyone."
        )
    return actions.get(
        (label, risk_level),
        "Exercise caution with this message."
    )


# ================================================================
#  REQUEST / RESPONSE SCHEMAS
# ================================================================
class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        example="WINNER ALERT! Your number won Rs.1 Lakh. Call 9876543210 to claim.",
    )


class BatchRequest(BaseModel):
    messages: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        example=[
            "Your OTP is 847291. Do not share. -SBI",
            "Click here to claim your free iPhone: http://prize-win.xyz",
        ],
    )


class PredictionResponse(BaseModel):
    prediction:       str
    spam_probability: float
    ham_probability:  float
    scam_score:       float
    risk_level:       str
    recommended_action: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total:   int
    spam_count: int
    ham_count:  int


class ModelInfoResponse(BaseModel):
    model_type:    str
    api_version:   str
    features:      str
    scam_score_info: str
    status:        str


# ================================================================
#  ENDPOINTS
# ================================================================
@app.get("/", tags=["Health"])
def home():
    return {
        "service": "ScamShield API",
        "version": "2.0.0",
        "model":   "XGBoost + TF-IDF (unigrams + bigrams)",
        "status":  "running" if MODEL else "model not loaded",
        "docs":    "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: MessageRequest):
    """
    Classify a single SMS message as spam or ham.

    Returns:
    - **prediction**: `spam` or `ham`
    - **spam_probability**: model confidence it is spam (0.0 – 1.0)
    - **ham_probability**: model confidence it is ham (0.0 – 1.0)
    - **scam_score**: risk score 0–100 (spam_probability × 100)
    - **risk_level**: `Safe` / `Low` / `Medium` / `High`
    - **recommended_action**: plain-language advice
    """
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are present.",
        )

    try:
        vec        = VECTORIZER.transform([data.message.strip()])
        pred       = int(MODEL.predict(vec)[0])
        label      = "spam" if pred == 1 else "ham"
        scam_score = get_scam_score(MODEL, vec)
        spam_prob  = round(scam_score / 100, 6)
        ham_prob   = round(1 - spam_prob, 6)
        risk_level = get_risk_level(scam_score)

        logger.info(
            f"predict | label={label} score={scam_score} "
            f"risk={risk_level} | msg={data.message[:60]}"
        )

        return PredictionResponse(
            prediction         = label,
            spam_probability   = spam_prob,
            ham_probability    = ham_prob,
            scam_score         = scam_score,
            risk_level         = risk_level,
            recommended_action = get_action(label, risk_level),
        )

    except Exception as e:
        logger.error(f"predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(data: BatchRequest):
    """
    Classify up to 100 messages in a single request (vectorised, fast).
    Ideal for bulk screening of message inboxes.
    """
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    try:
        cleaned = [m.strip() for m in data.messages]
        vecs    = VECTORIZER.transform(cleaned)

        if hasattr(MODEL, "predict_proba"):
            probs_all = MODEL.predict_proba(vecs)
            spam_probs = probs_all[:, 1]
            ham_probs  = probs_all[:, 0]
        else:
            spam_probs = []
            ham_probs  = []
            for i in range(vecs.shape[0]):
                s = get_scam_score(MODEL, vecs[i]) / 100
                spam_probs.append(s)
                ham_probs.append(1 - s)
            import numpy as np
            spam_probs = np.array(spam_probs)
            ham_probs  = np.array(ham_probs)

        results = []
        for msg, sp, hp in zip(cleaned, spam_probs, ham_probs):
            scam_score = round(float(sp) * 100, 2)
            label      = "spam" if sp >= 0.5 else "ham"
            risk_level = get_risk_level(scam_score)
            results.append(PredictionResponse(
                prediction         = label,
                spam_probability   = round(float(sp), 6),
                ham_probability    = round(float(hp), 6),
                scam_score         = scam_score,
                risk_level         = risk_level,
                recommended_action = get_action(label, risk_level),
            ))

        spam_count = sum(1 for r in results if r.prediction == "spam")
        return BatchPredictionResponse(
            results    = results,
            total      = len(results),
            spam_count = spam_count,
            ham_count  = len(results) - spam_count,
        )

    except Exception as e:
        logger.error(f"batch predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return metadata about the loaded model."""
    return ModelInfoResponse(
        model_type     = "XGBoost (XGBClassifier) + TF-IDF Vectorizer",
        api_version    = "2.0.0",
        features       = "TF-IDF unigrams + bigrams, max_features=10000, sublinear_tf=True",
        scam_score_info= (
            "scam_score = spam_probability × 100. "
            "Range 0–100. Risk bands: Safe(0-20), Low(21-49), Medium(50-79), High(80-100)."
        ),
        status = "loaded" if MODEL else "not loaded",
    )
