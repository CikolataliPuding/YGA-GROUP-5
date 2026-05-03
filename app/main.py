# app/main.py
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

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: PromptRequest, background_tasks: BackgroundTasks):
    result = classifier.classify(req.text)
    
    honeypot_session = None
    print(f"DEBUG decision: {result['decision']}")  # ← ekle
    if result["decision"] == "TEHDIT":
        print("DEBUG honeypot tetikleniyor")  # ← ekle
        honeypot_session = route_to_honeypot(req.text, background_tasks)
        print(f"DEBUG session_id: {honeypot_session}")  # ← ekle
    
    return ClassifyResponse(**result, honeypot_session=honeypot_session)

@app.get("/health")
def health():
    return {"status": "ok"}