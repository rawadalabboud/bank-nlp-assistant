# src/rag_build.py
"""
Build the RAG artifacts (FAISS index, metadata, config) from markdown FAQ files.

Usage:
    python src/rag_build.py
"""

from pathlib import Path
import re, json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------- Config ---------
FAQ_DIR = Path("faq")                      # input folder with *.md files
OUT_DIR = Path("models/rag")               # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FP = OUT_DIR / "index.faiss"
META_FP  = OUT_DIR / "meta.jsonl"
CONF_FP  = OUT_DIR / "config.json"

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_MAX_CHARS = 900  # ≈200–300 words
BATCH_SIZE = 64


# --------- Helpers ---------
def parse_front_matter(text: str) -> Dict[str, str]:
    """
    Extract YAML-style front matter from markdown (between --- lines).
    """
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) > 2:
            body = parts[2].strip()
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, body
    return meta, text


def split_by_headings(md: str) -> List[Dict[str, str]]:
    """
    Split markdown into sections by headings (# ...).
    """
    sections, current_h, buf = [], None, []
    for line in md.splitlines():
        if line.strip().startswith("#"):
            if buf and current_h:
                sections.append({"heading": current_h, "text": "\n".join(buf).strip()})
            current_h = line.strip("# ").strip()
            buf = []
        else:
            buf.append(line)
    if buf and current_h:
        sections.append({"heading": current_h, "text": "\n".join(buf).strip()})
    return sections


def sentence_chunk(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """
    Chunk text into pieces of ≈max_chars, splitting by sentences.
    """
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], []
    total = 0
    for s in sents:
        if total + len(s) > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf, total = [], 0
        buf.append(s); total += len(s)
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def build_corpus() -> List[Dict[str, str]]:
    corpus = []
    for f in FAQ_DIR.glob("*.md"):
        raw = f.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        sections = split_by_headings(body)
        for sec in sections:
            chunks = sentence_chunk(sec["text"], max_chars=CHUNK_MAX_CHARS)
            for c in chunks:
                corpus.append({
                    "text": c,
                    "heading": sec["heading"],
                    "category": meta.get("category"),
                    "source": meta.get("source"),
                    "file": f.name,
                })
    return corpus


# --------- Main ---------
def main():
    print("Loading corpus from", FAQ_DIR)
    corpus = build_corpus()
    print(f"Total passages: {len(corpus)}")

    print(f"Encoding with {EMB_MODEL_NAME} ...")
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts = [d["text"] for d in corpus]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=True)

    print("Building FAISS index ...")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))

    print("Saving artifacts ...")
    faiss.write_index(index, str(INDEX_FP))
    with META_FP.open("w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    CONF_FP.write_text(json.dumps({"model_name": EMB_MODEL_NAME}, indent=2), encoding="utf-8")

    print(f"Done. Saved {len(corpus)} passages to {OUT_DIR}")


if __name__ == "__main__":
    main()
