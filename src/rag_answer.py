# src/rag_answer.py
"""
RAG helpers: FAISS + MiniLM dense retrieval + CrossEncoder reranking (sigmoid),
strict on-topic stitching, and a guardrail threshold.

Build artifacts first:  python src/rag_build.py
Default API endpoint should call:  answer_with_rag_rerank
"""

from pathlib import Path
from typing import List, Dict, Any
import os
import json
import re
import numpy as np

# --- Dependencies with clear errors -------------------------------------------------
try:
    import faiss
except ImportError as e:
    raise RuntimeError("faiss is not installed. Install with: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    raise RuntimeError("sentence-transformers is not installed. Install with: pip install sentence-transformers") from e


def synthesize_short_answer(query: str, stitched_list: list) -> str:
    q = query.lower()
    if "atm" in q and ("swallow" in q or "retain" in q or "kept" in q or "confiscat" in q):
        return ("If an ATM kept your card: note the ATM location/time, contact the ATM owner and La Banque Postale support, "
                "place the card in opposition if you can’t recover it quickly, and request a replacement card.")
    return ""


# --- Paths & existence checks -------------------------------------------------------
RAG_DIR = Path("models/rag")
INDEX_FP = RAG_DIR / "index.faiss"
META_FP  = RAG_DIR / "meta.jsonl"
CONF_FP  = RAG_DIR / "config.json"

if not (INDEX_FP.exists() and META_FP.exists() and CONF_FP.exists()):
    raise RuntimeError("RAG artifacts not found. Run the build step first (e.g., `python src/rag_build.py`).")


# --- Config ------------------------------------------------------------------------
_cfg = json.loads(CONF_FP.read_text(encoding="utf-8"))

EMB_MODEL_NAME    = os.getenv("EMB_MODEL_NAME", _cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Guardrail threshold on *probability* after sigmoid (0..1). Tuned for your data.
RETRIEVE_THRESHOLD = float(os.getenv("RETRIEVE_THRESHOLD", "0.2"))

# Accept either RERANK_* (what the UI sets) or legacy RAG_* env names
DEFAULT_K_RETR  = int(os.getenv("RERANK_K_RETR",  os.getenv("RAG_K_RETR",  "30")))
DEFAULT_K_FINAL = int(os.getenv("RERANK_K_FINAL", os.getenv("RAG_K_FINAL", "5")))

# --- Load index, metadata, models ---------------------------------------------------
INDEX = faiss.read_index(str(INDEX_FP))
RETR  = SentenceTransformer(EMB_MODEL_NAME)     # dense embedder
RERANK = CrossEncoder(RERANK_MODEL_NAME)        # cross-encoder

META: List[Dict[str, Any]] = [
    json.loads(line) for line in META_FP.read_text(encoding="utf-8").splitlines() if line.strip()
]


# --- Optional: light keyword boosting for banking intents ---------------------------
BANK_RULES = [
    (r"\batm\b|\bcash\s?machine\b|\bgab\b|\bdistributor\b", " atm cash-machine gab atm "),
    (r"\b(swallowed|kept|retained|confiscated)\b.*\b(card)\b", " card swallowed retained kept atm "),
    (r"\blost\b|\bstolen\b", " opposition block hotlist card "),
    (r"\bbeneficiar(?:y|ies)\b|\badd recipient\b", " transfer beneficiary sepa iban "),
    (r"\bphishing\b|\bfraud\b", " phishing fraud scam suspicious email "),
]

def rewrite_query(q: str) -> str:
    extra = []
    for pat, add in BANK_RULES:
        if re.search(pat, q, flags=re.I):
            extra.append(add)
    return (q + " " + " ".join(extra)).strip()


# --- Core retrieval (dense + FAISS) ------------------------------------------------
def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    q = rewrite_query(query)
    qvec = RETR.encode([q], normalize_embeddings=True)
    D, I = INDEX.search(np.asarray(qvec, dtype="float32"), k)
    out: List[Dict[str, Any]] = []
    for score, idx in zip(D[0], I[0]):
        d = META[int(idx)]
        out.append({
            "score": float(score),
            "text": d["text"],
            "heading": d.get("heading"),
            "category": d.get("category"),
            "source": d.get("source"),
            "file": d.get("file"),
        })
    return out


# --- Utilities ---------------------------------------------------------------------
def _normalize(txt: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", txt.lower()).strip()

def dedup_passages(passages: List[Dict[str, Any]], min_words: int = 3) -> List[Dict[str, Any]]:
    """Remove near-duplicates by prefix of normalized text. Keeps first occurrence."""
    seen, out = set(), []
    for p in passages:
        norm = " ".join(_normalize(p["text"]).split()[:min_words])
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p)
    return out

ATM_RULE  = re.compile(r"\batm\b|\bgab\b|\bcash\s?machine\b", re.I)
SWAL_RULE = re.compile(r"\bswallow(?:ed)?\b|\bretained?\b|\bkept\b|\bconfiscated\b", re.I)

def is_on_topic(query: str, text: str) -> bool:
    q = query.lower()
    t = text.lower()
    if ATM_RULE.search(q) or SWAL_RULE.search(q):
        return bool(ATM_RULE.search(t) or SWAL_RULE.search(t))
    SPEC_KEYS = ["beneficiary", "iban", "password", "phishing", "fraud", "transfer"]
    return any(k in q and k in t for k in SPEC_KEYS)


# --- Reranked retrieval (CrossEncoder with sigmoid normalization) -------------------
def retrieve_rerank(query: str, k_retr: int = DEFAULT_K_RETR, k_final: int = DEFAULT_K_FINAL) -> List[Dict[str, Any]]:
    # allow env override here too
    k_retr  = int(os.getenv("RERANK_K_RETR",  k_retr))
    k_final = int(os.getenv("RERANK_K_FINAL", k_final))

    cand = retrieve(query, k=k_retr)
    if not cand:
        return []

    pairs = [(query, c["text"]) for c in cand]
    raw = RERANK.predict(pairs)                  # logits
    probs = 1.0 / (1.0 + np.exp(-raw))          # sigmoid -> [0,1]

    order = np.argsort(probs)[::-1][:k_final]
    out: List[Dict[str, Any]] = []
    for j in order:
        c = dict(cand[j])
        c["rerank_score_raw"] = float(raw[j])
        c["rerank_score"] = float(probs[j])
        out.append(c)
    return out


# --- Public answer helpers ---------------------------------------------------------
def answer_with_rag(query: str, k: int = DEFAULT_K_FINAL) -> Dict[str, Any]:
    hits = retrieve(query, k=k)
    stitched = "\n\n".join([f"- {h['text']}" for h in hits[:3]])
    srcs = [{"file": h["file"], "heading": h["heading"], "source": h["source"]} for h in hits[:3]]
    return {
        "query": query,
        "answer": f"Relevant policy snippets:\n\n{stitched}",
        "sources": srcs,
        "matches": hits,
    }


def answer_with_rag_rerank(query: str, k_retr: int = DEFAULT_K_RETR, k_final: int = DEFAULT_K_FINAL) -> Dict[str, Any]:
    # read UI overrides (so sliders always win)
    k_retr  = int(os.getenv("RERANK_K_RETR",  k_retr))
    k_final = int(os.getenv("RERANK_K_FINAL", k_final))

    # rerank exactly k_final (no hidden 5-cap)
    top = retrieve_rerank(query, k_retr=k_retr, k_final=max(k_final, 1))
    if not top:
        return {
            "query": query,
            "answer": "I couldn’t find a precise answer. Please rephrase your question or contact support (3639).",
            "sources": [],
            "matches": [],
        }

    # boost specific known FAQ if applicable
    if ATM_RULE.search(query) and SWAL_RULE.search(query):
        for t in top:
            if t.get("heading") and "card swallowed" in t["heading"].lower():
                t["rerank_score"] = max(t.get("rerank_score", 0.0), 0.95)
                break
        top = sorted(top, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    # dedup and guardrail
    top = dedup_passages(top)
    if float(top[0].get("rerank_score", 0.0)) < RETRIEVE_THRESHOLD:
        return {
            "query": query,
            "answer": "I couldn’t find a precise answer. Please rephrase your question or contact support (3639).",
            "sources": [],
            "matches": top[:k_final],
        }

    # strict on-topic stitch (up to 3 bullets just for readability)
    on_topic = [t for t in top if is_on_topic(query, t["text"])]
    stitched_list = (on_topic or top)[:3]

    matches = top[:k_final]

    short = synthesize_short_answer(query, stitched_list)
    stitched = "\n\n".join([f"- {t['text']}" for t in stitched_list])
    srcs = [{"file": t["file"], "heading": t["heading"], "source": t["source"]} for t in stitched_list]
    answer_text = (short + ("\n\n" if short else "") + f"Reranked (deduped) snippets:\n\n{stitched}")
    return {
        "query": query,
        "answer": answer_text,
        "sources": srcs,
        "matches": matches,
    }

def debug_rerank(query: str, k_retr: int = 30, k_final: int = 5):
    ranked = retrieve_rerank(query, k_retr=k_retr, k_final=k_final)
    return {
        "top_scores": [r.get("rerank_score", 0.0) for r in ranked],
        "top_headings": [r.get("heading", "") for r in ranked],
        "count": len(ranked),
    }
