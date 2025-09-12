from pathlib import Path
import joblib

MODEL = joblib.load(Path("models/intent/sklearn/model.joblib"))

def predict_intent(text: str):
    proba = MODEL.predict_proba([text])[0]
    idx = proba.argmax()
    pred = MODEL.classes_[idx]
    return {"intent": pred, "confidence": float(proba[idx])}
