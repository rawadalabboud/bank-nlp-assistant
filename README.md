# 💳 Bank NLP Assistant

A **Customer Support NLP Assistant** for banking use cases.  
It combines **intent classification** (using machine learning & transformers) with an API ready for deployment.  
Future steps will extend it with **RAG** (Retrieval-Augmented Generation).

---

## 🚀 Project Overview

This project builds an NLP pipeline for banking customer support:
1. **Data**: Banking77 dataset (10k queries, 77 intents).
2. **Baseline Model**: TF-IDF + Logistic Regression (sklearn).
3. **Transformer Model**: Fine-tuned DistilBERT with Hugging Face.
4. **API**: FastAPI endpoint to classify customer queries in real time.

---

## 📂 Repository Structure

```
bank-nlp-assistant/
│
├── api/                     # FastAPI service
│   └── main.py              # /health and /classify endpoints
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_baseline.ipynb    # Baseline: sklearn classifier
│   └── 02_transformer_baseline.ipynb  # DistilBERT fine-tuning
│
├── src/                     # Source code
│   ├── data_banking77.py    # Dataset preparation (Banking77 → CSVs)
│   ├── train_intent_sklearn.py  # Train sklearn baseline
│   ├── infer_intent.py      # Inference with sklearn baseline
│   └── infer_intent_transformer.py  # Inference with DistilBERT
│
├── models/                  # Saved models
│   └── intent/transformer/  # DistilBERT configs + tokenizer (no large weights)
│
├── data/processed/          # Train/val/test splits
│
├── requirements.txt         # Project dependencies
├── LICENSE
└── README.md
```

---

## 📊 Results

### Baseline (sklearn)
- Accuracy: ~70%
- Macro-F1: ~0.68
- Clear confusion between similar intents (e.g. *“card not working”* vs *“card acceptance”*).

### Transformer (DistilBERT)
- Accuracy: **82%**
- Macro-F1: **0.80**
- Handles subtle differences much better.
- Visualizations:
  - Confusion matrix (top 20 intents).
  - Misclassification analysis.
  - Confidence distribution.

---

## 🔧 Running Locally

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run notebooks
Open Jupyter or VSCode to explore:
- `01_baseline.ipynb`
- `02_transformer_baseline.ipynb`

### 3. Start API
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Test endpoints
```bash
# health check
curl http://127.0.0.1:8000/health

# classify a query
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"I lost my bank card yesterday, what should I do?"}'
```

---

## 🔮 Next Steps (planned)
- [ ] Add **RAG module** (FAQ search).
- [ ] Confidence-based fallback (classifier + RAG).
- [ ] Dockerize API for easy deployment.
- [ ] Add monitoring + evaluation scripts.

---

## 📜 License
MIT License.
