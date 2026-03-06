from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

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
    allow_origins=["*"],  # hackathon/demo setting
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "ScamShield API running"}

@app.post("/predict")
def predict(data: Message):
    vec = vectorizer.transform([data.message])
    pred = model.predict(vec)[0]
    label = "spam" if pred == 1 else "ham"
    return {"prediction": label}
