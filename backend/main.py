import os
from pathlib import Path
from functools import lru_cache
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Clasificador de tickets - Fase II", version="1.0.0")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/ticket_pipeline.joblib"))
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))

class Ticket(BaseModel):
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=10000)

@lru_cache(maxsize=1)
def model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no disponible: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.get("/")
def index():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.is_file()}

@app.post("/predict")
def predict(ticket: Ticket):
    text = (ticket.subject.strip() + " " + ticket.body.strip()).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Texto vacío")
    try:
        clf = model()
        probabilities = clf.predict_proba([text])[0]
        idx = int(probabilities.argmax())
        confidence = float(probabilities[idx])
        return {"queue": str(clf.classes_[idx]), "confidence": round(confidence, 4),
                "requires_human_review": confidence < THRESHOLD, "threshold": THRESHOLD}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
