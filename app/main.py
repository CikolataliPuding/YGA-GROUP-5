# app/main.py
import time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from app.classifier import ErlikClassifier
from honeypot.router import route_to_honeypot

app = FastAPI(title="ErlikGate Karar Motoru", version="1.0")
classifier = ErlikClassifier()

class PromptRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    decision:         str
    label:            str
    confidence:       float
    inference_ms:     float
    source:           str
    rule_match:       str | None
    honeypot_session: str | None = None
    e2e_ms:           float = 0.0

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: PromptRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    result = classifier.classify(req.text)

    honeypot_session = None
    if result["decision"] == "TEHDIT":
        honeypot_session = route_to_honeypot(req.text, background_tasks)

    e2e_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ClassifyResponse(**result, honeypot_session=honeypot_session, e2e_ms=e2e_ms)

@app.get("/health")
def health():
    return {"status": "ok"}