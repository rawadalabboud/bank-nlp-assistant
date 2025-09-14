# src/rag_eval.py
"""
Evaluate RAG retrieval quality (Hit@3, MRR@10) on a small set of queries.

Usage:
    python src/rag_eval.py
"""

import numpy as np
from typing import List, Dict
from src.rag_answer import answer_with_rag, answer_with_rag_rerank


# --------- Define evaluation set ---------
# Add more queries here to expand coverage
EVAL: List[Dict[str, any]] = [
    {"q": "ATM swallowed my card, what should I do?", "must_contain": ["atm", "swallowed", "card"]},
    {"q": "How can I add a transfer beneficiary?", "must_contain": ["beneficiary", "iban"]},
    {"q": "I forgot my online banking password", "must_contain": ["password", "login"]},
    {"q": "How to recognize a phishing email?", "must_contain": ["phishing", "fraud", "email"]},
]


# --------- Helpers ---------
def contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def eval_hit_mrr(fn, dataset, k=10):
    hits, rr = [], []
    for item in dataset:
        cand = fn(item["q"], k_retr=k, k_final=min(5, k))["matches"] if "rerank" in fn.__name__ else answer_with_rag(item["q"], k=k)["matches"]
        found_rank = None
        for i, c in enumerate(cand):
            if contains_any(c["text"], item["must_contain"]):
                found_rank = i + 1
                break
        hits.append(1 if found_rank and found_rank <= 3 else 0)
        rr.append(1.0 / found_rank if found_rank else 0.0)
    return {"Hit@3": float(np.mean(hits)), "MRR@10": float(np.mean(rr))}


# --------- Run evaluation ---------
if __name__ == "__main__":
    print("Evaluating Base (dense only)...")
    base_metrics = eval_hit_mrr(answer_with_rag, EVAL, k=10)
    print("Base:", base_metrics)

    print("\nEvaluating Rerank (cross-encoder)...")
    rerank_metrics = eval_hit_mrr(answer_with_rag_rerank, EVAL, k=10)
    print("Rerank:", rerank_metrics)
