# ================================================================
#  ScamShield — FastAPI Backend
#  Model: XGBoost + TF-IDF  |  Heuristic booster for edge cases
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
import re
import math
import pickle
import logging
from typing import List

import numpy as np
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
        "Detects SMS scam/spam messages using XGBoost + TF-IDF "
        "with a heuristic booster for common scam patterns. "
        "Returns prediction, probabilities, scam_score (0-100), "
        "risk level, and recommended action."
    ),
    version="2.1.0",
)

# Allow your deployed frontend origins; fall back to permissive for dev
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "https://scam-shield-three.netlify.app,http://localhost:3000,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ================================================================
#  HEURISTIC BOOSTER
#  The ML model is trained on a small (500-row) synthetic dataset,
#  so it misses many real-world scam patterns. These rules detect
#  obvious spam signals and return a heuristic scam probability
#  that is blended with the model output.
# ================================================================

# Compiled once at import time for performance
_SCAM_PATTERNS = [
    (re.compile(r"\b(won|winner|selected|lucky\s*draw|prize|reward)\b", re.I), 0.35),
    (re.compile(r"\bclaim\b.*\b(now|today|immediately|before)\b", re.I), 0.30),
    (re.compile(r"\bfree\s+(iphone|gift|voucher|cash|card)\b", re.I), 0.40),
    (re.compile(r"\b(click\s+here|tap\s+here|visit\s+now)\b", re.I), 0.25),
    (re.compile(r"https?://\S+\.(xyz|win|top|buzz|club|info|tk|ml|ga|cf)\b", re.I), 0.40),
    (re.compile(r"\b(kyc|pan|aadhaar)\b.*\b(expir|updat|verif|block|suspend)", re.I), 0.40),
    (re.compile(r"\b(sim|account|card)\b.*\b(block|suspend|deactivat)", re.I), 0.30),
    (re.compile(r"\bcall\s+\d{10}\b", re.I), 0.20),
    (re.compile(r"\bclick\b.*\blink\b", re.I), 0.25),
    (re.compile(r"\b(earn|make)\s+rs\.?\s*\d", re.I), 0.30),
    (re.compile(r"\b(work\s+from\s+home|part\s*time\s+job)\b", re.I), 0.30),
    (re.compile(r"\b(congratulations|congrats)\b", re.I), 0.25),
    (re.compile(r"rs\.?\s*\d[\d,]*\s*(lakh|crore|gift|cash|prize)", re.I), 0.30),
    (re.compile(r"\b(urgent|immediately|final\s*notice|last\s*chance|expires?\s*today)\b", re.I), 0.20),
    (re.compile(r"\bdo\s+not\s+ignore\b", re.I), 0.15),
    (re.compile(r"\b(send|share)\s+(your|ur)\s+(details?|info|otp|pin)\b", re.I), 0.35),
    (re.compile(r"\b(compromised|hacked|unauthorized|suspicious\s+activity)\b", re.I), 0.30),
    (re.compile(r"\bfree\b.*\b(gift|offer|prize|reward)\b", re.I), 0.30),
    (re.compile(r"\bloan\b.*\b(approved|apply\s*now)\b", re.I), 0.30),
    (re.compile(r"\b(flat|upto|up\s*to)\s*\d{2,3}%\s*off\b", re.I), 0.30),
    (re.compile(r"\bshare\b.*\bwith\b.*\bfriends\b", re.I), 0.30),
    (re.compile(r"\bwin\b.*\b(car|cash|prize|phone|iphone)\b", re.I), 0.40),
    (re.compile(r"\blimited\s*period\s*offer\b", re.I), 0.25),
    (re.compile(r"\bshop\s*now\b", re.I), 0.15),
]

# Patterns that strongly indicate ham — suppress false positives
_HAM_PATTERNS = [
    (re.compile(r"\byour\s+otp\s+(is|for|:)\b", re.I), -0.30),
    (re.compile(r"\bdo\s+not\s+share\b.*otp", re.I), -0.25),
    (re.compile(r"\border\s+#?\w+\s+(has\s+been\s+)?(shipped|delivered|dispatched)\b", re.I), -0.25),
    (re.compile(r"\b(debited|credited)\s+from\s+a/c\b", re.I), -0.25),
    (re.compile(r"\bavl\.?\s*bal", re.I), -0.20),
    (re.compile(r"\bbill\b.*\bdue\b.*\bofficial\s+app\b", re.I), -0.20),
    (re.compile(r"\binterview\b", re.I), -0.15),
]


def heuristic_scam_score(text: str) -> float:
    """
    Return a heuristic adjustment score in [-0.30, ~1.0].
    Positive = spammy, negative = ham-like.
    Values are capped and combined additively.
    """
    score = 0.0
    for pattern, weight in _SCAM_PATTERNS:
        if pattern.search(text):
            score += weight
    for pattern, weight in _HAM_PATTERNS:
        if pattern.search(text):
            score += weight  # weight is already negative
    return max(-0.30, min(score, 1.0))


def blend_scores(model_spam_prob: float, heuristic: float) -> float:
    """
    Blend model probability with heuristic signal.

    The ML model is trained on only 500 synthetic samples, so it
    misses many real-world scam patterns. When the heuristic detects
    scam signals, we trust it over the weak model.
    """
    if heuristic < -0.10:
        # Strong ham signal (e.g. OTP message) — suppress model FP
        return max(0.0, model_spam_prob * 0.3)

    if heuristic >= 0.50:
        # Very strong scam signal — use heuristic directly
        return min(heuristic, 1.0)

    if heuristic >= 0.25:
        # Moderate scam signal — take the max of both
        return max(model_spam_prob, min(heuristic, 1.0))

    # Mild or no heuristic signal — trust the model mostly
    return max(model_spam_prob, min(heuristic, 1.0) * 0.5)


# ================================================================
#  MODEL LOADING
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
        f"Could not find '{filename}'. Searched: {search_paths}"
    )


def load_artifacts():
    """Load XGBoost model + TF-IDF vectorizer from disk."""
    model_path = find_file("xgb_spam_model.pkl")
    tfidf_path = find_file("tfidf_vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info(f"Model loaded from   : {model_path}")
    logger.info(f"Vectorizer loaded from: {tfidf_path}")
    return model, vectorizer


try:
    MODEL, VECTORIZER = load_artifacts()
    logger.info("ScamShield model ready.")
except FileNotFoundError as e:
    logger.error(str(e))
    MODEL, VECTORIZER = None, None


# ================================================================
#  HELPER FUNCTIONS
# ================================================================
def get_model_spam_prob(model, vec) -> float:
    """
    Extract spam probability from any sklearn-compatible model.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        return float(probs[1])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        return 1.0 / (1.0 + math.exp(-score))

    return 1.0 if int(model.predict(vec)[0]) == 1 else 0.0


def get_risk_level(scam_score: float) -> str:
    if scam_score >= 80:
        return "High"
    elif scam_score >= 50:
        return "Medium"
    elif scam_score >= 21:
        return "Low"
    return "Safe"


def get_action(label: str, risk_level: str) -> str:
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
    return actions.get((label, risk_level), "Exercise caution with this message.")


def classify_message(text: str) -> dict:
    """
    Core classification: blend ML model + heuristic booster.
    Returns dict with prediction, probabilities, scam_score, risk_level, action.
    """
    text = text.strip()
    vec = VECTORIZER.transform([text])

    model_prob = get_model_spam_prob(MODEL, vec)
    h_score = heuristic_scam_score(text)
    final_prob = blend_scores(model_prob, h_score)

    scam_score = round(final_prob * 100, 2)
    label = "spam" if final_prob >= 0.50 else "ham"
    risk_level = get_risk_level(scam_score)

    return {
        "prediction": label,
        "spam_probability": round(final_prob, 6),
        "ham_probability": round(1 - final_prob, 6),
        "scam_score": scam_score,
        "risk_level": risk_level,
        "recommended_action": get_action(label, risk_level),
    }


# ================================================================
#  REQUEST / RESPONSE SCHEMAS
# ================================================================
class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["WINNER ALERT! Your number won Rs.1 Lakh. Call 9876543210 to claim."],
    )


class BatchRequest(BaseModel):
    messages: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=[
            [
                "Your OTP is 847291. Do not share. -SBI",
                "Click here to claim your free iPhone: http://prize-win.xyz",
            ]
        ],
    )


class PredictionResponse(BaseModel):
    prediction: str
    spam_probability: float
    ham_probability: float
    scam_score: float
    risk_level: str
    recommended_action: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    spam_count: int
    ham_count: int


class ModelInfoResponse(BaseModel):
    model_type: str
    api_version: str
    features: str
    scam_score_info: str
    status: str


# ================================================================
#  ENDPOINTS
# ================================================================
@app.get("/", tags=["Health"])
def home():
    return {
        "service": "ScamShield API",
        "version": "2.1.0",
        "model": "XGBoost + TF-IDF + Heuristic Booster",
        "status": "running" if MODEL else "model not loaded",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: MessageRequest):
    """
    Classify a single SMS message as spam or ham.

    Uses ML model + heuristic pattern matching for robust detection.
    """
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are present.",
        )

    try:
        result = classify_message(data.message)
        logger.info(
            f"predict | label={result['prediction']} "
            f"score={result['scam_score']} risk={result['risk_level']} "
            f"| msg={data.message[:60]}"
        )
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(data: BatchRequest):
    """
    Classify up to 100 messages in a single request.
    """
    if MODEL is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        results = [
            PredictionResponse(**classify_message(msg))
            for msg in data.messages
        ]
        spam_count = sum(1 for r in results if r.prediction == "spam")
        return BatchPredictionResponse(
            results=results,
            total=len(results),
            spam_count=spam_count,
            ham_count=len(results) - spam_count,
        )

    except Exception as e:
        logger.error(f"batch predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    return ModelInfoResponse(
        model_type="XGBoost + TF-IDF + Heuristic Booster",
        api_version="2.1.0",
        features=(
            "TF-IDF unigrams + bigrams (max_features=10000, sublinear_tf=True), "
            "plus regex-based heuristic booster for common Indian scam patterns"
        ),
        scam_score_info=(
            "scam_score = blended(model_prob, heuristic) × 100. "
            "Range 0–100. Risk bands: Safe(0-20), Low(21-49), Medium(50-79), High(80-100)."
        ),
        status="loaded" if MODEL else "not loaded",
    )
