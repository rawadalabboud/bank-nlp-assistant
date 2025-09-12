# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.infer_intent_transformer import predict_intent_transformer

app = FastAPI(title="Bank NLP Assistant — Intent Classifier")

class Query(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "distilbert-transformer"}

@app.post("/classify")
def classify(q: Query):
    # return top-5 intents with confidences
    return predict_intent_transformer(q.text, k=5)
