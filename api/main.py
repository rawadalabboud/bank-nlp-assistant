# api/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel

from src.infer_intent_transformer import predict_intent_transformer
from src.rag_answer import (
    answer_with_rag,              # baseline: dense only
    answer_with_rag_rerank,       # default: dense + cross-encoder + dedup + guardrail + on-topic stitch
    retrieve_rerank,              # for /answer_debug
    RETRIEVE_THRESHOLD,           # expose threshold value
)

app = FastAPI(title="Bank NLP Assistant")

# --------- Config (via env) ----------
INTENT_TOPK = int(os.getenv("INTENT_TOPK", "5"))
RERANK_K_RETR = int(os.getenv("RERANK_K_RETR", "20"))
RERANK_K_FINAL = int(os.getenv("RERANK_K_FINAL", "5"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.30"))  # for smart routing


class Query(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "intent_model": "distilbert-transformer",
        "rag_retriever": "sentence-transformers (MiniLM)",
        "rag_reranker": os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "rerank_k_retr": RERANK_K_RETR,
        "rerank_k_final": RERANK_K_FINAL,
    }


@app.post("/classify")
def classify(q: Query):
    """Intent classification (top-K) with confidences."""
    return predict_intent_transformer(q.text, k=INTENT_TOPK)


@app.post("/answer")
def answer(q: Query):
    """RAG answer with cross-encoder reranking (default)."""
    return answer_with_rag_rerank(q.text, k_retr=RERANK_K_RETR, k_final=RERANK_K_FINAL)


@app.post("/answer_base")
def answer_base(q: Query):
    """Baseline RAG (dense only) — for comparison & debugging."""
    return answer_with_rag(q.text, k=max(5, RERANK_K_FINAL))


@app.post("/smart-answer")
def smart_answer(q: Query):
    """
    Simple router:
      - If intent confidence < CONF_THRESHOLD → RAG-only
      - Else → intent + RAG (reranked)
    """
    clf = predict_intent_transformer(q.text, k=INTENT_TOPK)
    best = clf.get("best", {"confidence": 0.0})
    if float(best.get("confidence", 0.0)) < CONF_THRESHOLD:
        rag = answer_with_rag_rerank(q.text, k_retr=RERANK_K_RETR, k_final=RERANK_K_FINAL)
        return {"mode": "rag_fallback", "classification": clf, **rag}
    rag = answer_with_rag_rerank(q.text, k_retr=RERANK_K_RETR, k_final=RERANK_K_FINAL)
    return {"mode": "intent_plus_rag", "classification": clf, **rag}


@app.post("/answer_debug")
def answer_debug(q: Query):
    """Return top rerank scores & headings to debug threshold/ordering."""
    top = retrieve_rerank(q.text, k_retr=max(20, RERANK_K_RETR), k_final=max(10, RERANK_K_FINAL))
    return {
        "threshold": RETRIEVE_THRESHOLD,
        "top_scores": [float(t.get("rerank_score", 0.0)) for t in top[:5]],
        "top_headings": [t.get("heading") for t in top[:5]],
        "matches": top[:5],
    }
