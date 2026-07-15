# Bank NLP Assistant

Customer-support NLP stack: **intent classification** (Banking77), **RAG** over a Markdown FAQ, **cross-encoder reranking**, **FastAPI** backend, and **Streamlit** demo UI.

> Personal RAG + NLP project aligned with my production GenAI work. Portfolio: [rawad-portfolio](https://github.com/rawadalabboud/rawad-portfolio).

## Features

| Layer | Implementation | Notes |
|-------|----------------|-------|
| Intent (baseline) | TF-IDF + Logistic Regression | ~85% accuracy / macro-F1 |
| Intent (transformer) | DistilBERT fine-tuned | ~90% accuracy / macro-F1 |
| Retrieval | SentenceTransformers + FAISS | MiniLM embeddings |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Precision-focused FAQ answers |
| API | FastAPI | `/health`, `/classify`, `/answer`, `/answer_debug` |
| UI | Streamlit | Intent tab + FAQ assistant + debug view |

## Project structure

```
bank-nlp-assistant/
├── api/main.py              # FastAPI service
├── app.py                   # Streamlit demo
├── src/
│   ├── data_banking77.py
│   ├── train_intent_sklearn.py
│   ├── infer_intent.py
│   ├── infer_intent_transformer.py
│   ├── rag_build.py
│   ├── rag_eval.py
│   └── rag_answer.py
├── notebooks/               # 01 baseline → 03 RAG experiments
├── faq/                     # Markdown knowledge base
├── models/                  # Saved classifiers + FAISS index
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/rawadalabboud/bank-nlp-assistant.git
cd bank-nlp-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Streamlit UI

```bash
streamlit run app.py
```

### FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Classify: `POST /classify` with `{"text": "...", "k": 5}`  
- RAG answer: `POST /answer` with `{"text": "..."}`  

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_baseline.ipynb` | sklearn TF-IDF + logistic regression |
| `02_transformer_baseline.ipynb` | DistilBERT fine-tuning |
| `03_rag.ipynb` | FAISS retrieval + rerank + guardrails |

## Tech stack

Python 3.9+ · Hugging Face Transformers · scikit-learn · PyTorch · FAISS · FastAPI · Streamlit

## Roadmap

- Docker + CI/CD deployment  
- RAG metrics (Hit@k, MRR) in eval harness  
- Optional LLM generation layer on top of reranked context  

## Author

**Rawad Al Abboud** — ML/AI Engineer · Paris  

- Portfolio: [github.com/rawadalabboud/rawad-portfolio](https://github.com/rawadalabboud/rawad-portfolio)  
- GitHub: [@rawadalabboud](https://github.com/rawadalabboud)  
- LinkedIn: [rawad-al-abboud](https://www.linkedin.com/in/rawad-al-abboud/)

## License

See [LICENSE](LICENSE).
