# Bank NLP Assistant

Customer Support NLP Assistant using intent classification, RAG, and LLMs.  
Built with **Hugging Face Transformers**, **Scikit-learn**, and **FastAPI** for deployment.

---

## 📂 Project Structure
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

## 🚀 Current Progress

### 1. Baseline (Notebook 01)
- Model: **TF-IDF + Logistic Regression**
- Accuracy: ~0.85
- Macro-F1: ~0.85

This baseline provides a strong starting point using classic ML methods.

### 2. Transformer Fine-Tuning (Notebook 02)
- Model: **DistilBERT fine-tuned on Banking77**
- Accuracy: ~0.90
- Macro-F1: ~0.90

DistilBERT outperforms the baseline, confirming that contextual embeddings capture user intent better than bag-of-words.

---

## ✅ Next Steps
- Add **error analysis** with confusion matrices for transformers.
- Experiment with **hyperparameters and larger models**.
- Implement **RAG (Retrieval-Augmented Generation)** in Notebook 03.
- Deploy the transformer model in the **FastAPI API**.

---

## 🛠️ Tech Stack
- **Python** (3.9+)
- **Transformers (Hugging Face)**
- **Scikit-learn**
- **Datasets (Hugging Face)**
- **PyTorch**
- **FastAPI + Uvicorn**
