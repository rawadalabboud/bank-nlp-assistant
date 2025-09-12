import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Paths
DATA = Path("data/processed")
MODELS = Path("models/intent/sklearn"); MODELS.mkdir(parents=True, exist_ok=True)

# Load CSVs
train = pd.read_csv(DATA/"train.csv")
val   = pd.read_csv(DATA/"val.csv")
test  = pd.read_csv(DATA/"test.csv")

# Build pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
])

# Train
pipe.fit(train.text, train.intent)

# Evaluate
print("Validation report:\n",
      classification_report(val.intent, pipe.predict(val.text), zero_division=0))
print("Test report:\n",
      classification_report(test.intent, pipe.predict(test.text), zero_division=0))

# Save
joblib.dump(pipe, MODELS/"model.joblib")
print("Saved model to", MODELS/"model.joblib")
