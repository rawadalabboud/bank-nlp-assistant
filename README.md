# 💳 Bank NLP Assistant

Customer Support NLP Assistant using **intent classification**, **retrieval-augmented generation (RAG)**, and **cross-encoder reranking**.  
Built with **Hugging Face Transformers**, **Scikit-learn**, **FastAPI**, and a **Streamlit demo UI**.  

---

## 📂 Project Structure
```
bank-nlp-assistant/
│
├── api/                     # FastAPI service (API endpoints)
│   └── main.py              # /health, /classify, /answer endpoints
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_baseline.ipynb             # Baseline: sklearn classifier
│   ├── 02_transformer_baseline.ipynb # DistilBERT fine-tuning
│   └── 03_rag.ipynb                  # RAG + rerank experiments
│
├── src/                     # Source code
│   ├── data_banking77.py           # Dataset preparation (Banking77 → CSVs)
│   ├── train_intent_sklearn.py     # Train sklearn baseline
│   ├── infer_intent.py             # Inference with sklearn baseline
│   ├── infer_intent_transformer.py # Inference with DistilBERT
│   ├── rag_build.py                # Build FAISS index + metadata
│   ├── rag_eval.py                 # Evaluate RAG retrieval + rerank
│   └── rag_answer.py               # RAG + rerank pipeline
│
├── models/                  # Saved models + FAISS artifacts
│   ├── intent/transformer/         # DistilBERT configs + tokenizer
│   └── rag/                        # FAISS index + metadata
│
├── faq/                     # Markdown FAQ knowledge base
│
├── app.py                   # Streamlit demo app
├── data/processed/          # Train/val/test splits
├── requirements.txt         # Project dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Current Progress

### 1. Baseline (Notebook 01)
- **Model**: TF-IDF + Logistic Regression  
- **Accuracy**: ~0.85 | **Macro-F1**: ~0.85  
➡️ Provides a strong classical ML baseline.  

### 2. Transformer Fine-Tuning (Notebook 02)
- **Model**: DistilBERT fine-tuned on Banking77  
- **Accuracy**: ~0.90 | **Macro-F1**: ~0.90  
➡️ Contextual embeddings capture user intent better than bag-of-words.  

### 3. RAG + Rerank (Notebook 03)
- **Retriever**: SentenceTransformers MiniLM + FAISS  
- **Reranker**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`)  
- **Guardrails**: Deduplication, thresholding, on-topic filtering  
➡️ Improves precision and provides explainable FAQ answers.  

### 4. API + Streamlit UI
- **FastAPI** backend with `/classify`, `/answer`, `/answer_debug`  
- **Streamlit** UI with:  
  - Intent classification tab (Top-K)  
  - FAQ Assistant with reranked snippets  
  - Debugging view of scores and sources  

---

## ✅ Next Steps
- Improve UI styling (dark/light themes, better highlighting).  
- Add **evaluation metrics** for RAG (Hit@k, MRR).  
- Experiment with **larger cross-encoders** and **LLM generation**.  
- Deploy with **Docker + CI/CD**.  

---

## 🛠️ Tech Stack
- **Python** (3.9+)  
- **Transformers (Hugging Face)**  
- **Scikit-learn**  
- **Datasets (Hugging Face)**  
- **PyTorch**  
- **FAISS**  
- **FastAPI + Uvicorn**  
- **Streamlit**  
