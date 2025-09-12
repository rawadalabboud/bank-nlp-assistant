# src/infer_intent_transformer.py
from pathlib import Path
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = Path("models/intent/transformer")
DEVICE = torch.device("cpu")  # keep CPU to avoid macOS MPS issues

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()

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
        "best": {"intent": classes[idxs[0]], "confidence": float(probs[idxs[0]])}
    }
