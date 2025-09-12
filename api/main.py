# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from src.infer_intent_transformer import predict_intent_transformer

MODEL_DIR = os.getenv("MODEL_DIR", "models/intent/transformer_ft")

app = FastAPI(title="Bank NLP Assistant — Intent Classifier")

class Query(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "distilbert-transformer", "model_dir": MODEL_DIR}

@app.post("/classify")
def classify(q: Query):
    return predict_intent_transformer(q.text, k=5)
