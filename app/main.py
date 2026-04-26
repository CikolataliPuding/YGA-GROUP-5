from fastapi import FastAPI
from pydantic import BaseModel
from app.classifier import ErlikClassifier

app = FastAPI(title="ErlikGate Karar Motoru", version="1.0")
classifier = ErlikClassifier()

class PromptRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    decision:     str
    label:        str
    confidence:   float
    inference_ms: float
    source:       str
    rule_match:   str | None

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: PromptRequest):
    result = classifier.classify(req.text)
    return ClassifyResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}