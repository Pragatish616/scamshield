from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import math

app = FastAPI(title="ScamShield Spam Detector API")

def find_model():
    if os.path.exists("scamshield/spam_svm_tfidf.joblib"):
        return "scamshield/spam_svm_tfidf.joblib"

    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "spam_svm_tfidf.joblib":
                return os.path.join(root, f)

model_path = find_model()

bundle = joblib.load(model_path)
vectorizer = bundle["vectorizer"]
model = bundle["model"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

def get_risk_level(score: float) -> str:
    if score >= 80:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def get_solution(label: str) -> str:
    if label == "spam":
        return (
            "Do not click any links, do not share personal or banking details, "
            "block the sender, and verify the message through the official source."
        )
    else:
        return (
            "This message appears relatively safe, but still avoid clicking unknown links "
            "and verify sensitive requests before responding."
        )

def get_risk_percentage(vec) -> float:
    # Case 1: model supports probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        return round(float(probs[1]) * 100, 2)

    # Case 2: fallback for SVM decision function
    elif hasattr(model, "decision_function"):
        decision = float(model.decision_function(vec)[0])
        probability_like = 1 / (1 + math.exp(-decision))
        return round(probability_like * 100, 2)

    # Case 3: fallback if neither exists
    return 100.0 if model.predict(vec)[0] == 1 else 0.0

@app.get("/")
def home():
    return {"message": "ScamShield API running"}

@app.post("/predict")
def predict(data: Message):
    vec = vectorizer.transform([data.message])
    pred = model.predict(vec)[0]

    label = "spam" if pred == 1 else "ham"
    risk_percentage = get_risk_percentage(vec)
    risk_factor = get_risk_level(risk_percentage)
    solution = get_solution(label)

    return {
        "prediction": label,
        "risk_percentage": risk_percentage,
        "risk_factor": risk_factor,
        "solution": solution
    }
