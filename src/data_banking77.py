from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

ds = load_dataset("banking77")  # has 'train' and 'test'
labels = ds["train"].features["label"].names

def to_df(split):
    df = pd.DataFrame(split)
    df["intent"] = df["label"].apply(lambda i: labels[i])
    return df[["text", "intent"]]

train_df = to_df(ds["train"])
test_df  = to_df(ds["test"])

# build a 10% validation split from train
val_df = train_df.sample(frac=0.10, random_state=42)
train_df = train_df.drop(val_df.index)

train_df.to_csv(OUT/"train.csv", index=False)
val_df.to_csv(OUT/"val.csv", index=False)
test_df.to_csv(OUT/"test.csv", index=False)

print("Saved CSVs to", OUT)