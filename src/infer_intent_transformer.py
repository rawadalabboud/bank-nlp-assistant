# src/infer_intent_transformer.py
from pathlib import Path
import os, numpy as np, torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Allow overriding the model path via env var
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/intent/transformer_ft"))
DEVICE = torch.device("cpu")  # keep CPU to avoid macOS MPS quirks

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

id2label = model.config.id2label
classes = [id2label[i] for i in range(len(id2label))]

def predict_intent_transformer(text: str, k: int = 5):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        probs = model(**enc).logits.softmax(dim=-1).cpu().numpy().squeeze()
    idxs = np.argsort(probs)[::-1][:k]
    return {
        "top_k": [{"intent": classes[i], "confidence": float(probs[i])} for i in idxs],
        "best": {"intent": classes[idxs[0]], "confidence": float(probs[idxs[0]])},
        "model_dir": str(MODEL_DIR)
    }
