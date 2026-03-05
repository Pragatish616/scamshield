from fastapi import FastAPI

app = FastAPI(
    title="ScamShield",
    description="Paste a message to check if it's Spam/Scam.",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
def home():
    return {"message": "ScamShield API running"}
