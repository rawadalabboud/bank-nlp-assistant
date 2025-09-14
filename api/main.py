# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.infer_intent_transformer import predict_intent_transformer
from src.rag_answer import (
    answer_with_rag_rerank,
    debug_rerank,              # <-- new small helper
    DEFAULT_K_RETR,
    DEFAULT_K_FINAL,
)

app = FastAPI(title="Bank NLP Assistant — API")

# ---------- Schemas ----------
class ClassifyQuery(BaseModel):
    text: str
    k: Optional[int] = 5       # client can choose top-k intents

class RAGQuery(BaseModel):
    text: str
    k_retr: Optional[int] = None
    k_final: Optional[int] = None

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": "distilbert-transformer", "rag": True}

# ---------- Intent ----------
@app.post("/classify")
def classify(q: ClassifyQuery):
    k = q.k or 5
    return predict_intent_transformer(q.text, k=k)

# ---------- RAG ----------
@app.post("/answer")
def answer(q: RAGQuery):
    k_retr = q.k_retr or DEFAULT_K_RETR
    k_final = q.k_final or DEFAULT_K_FINAL
    return answer_with_rag_rerank(q.text, k_retr=k_retr, k_final=k_final)

@app.post("/answer_debug")
def answer_debug(q: RAGQuery):
    k_retr = q.k_retr or DEFAULT_K_RETR
    k_final = q.k_final or DEFAULT_K_FINAL
    dbg = debug_rerank(q.text, k_retr=k_retr, k_final=k_final)
    dbg.update({"k_retr": k_retr, "k_final": k_final})
    return dbg
