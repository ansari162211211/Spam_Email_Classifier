"""Quick health check for the spam classifier."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from spam_classifier import (
    DATA_PATH,
    MODEL_PATH,
    load_dataset,
    load_model,
    predict_message,
)

print("=== 1. Files ===")
for p in [MODEL_PATH, DATA_PATH]:
    print(f"  {p.name}:", "OK" if p.exists() else "MISSING")

print("\n=== 2. Dataset ===")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}")
print("  Labels:", df["label"].value_counts().to_dict())

print("\n=== 3. Saved model ===")
pipe, name = load_model()
print(f"  Model type: {name}")
print(f"  Pipeline steps: {list(pipe.named_steps.keys())}")

print("\n=== 4. Hold-out test (20% stratified) ===")
messages, labels = load_dataset()
_, X_test, _, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)
y_pred = [predict_message(pipe, m)[0] for m in X_test]
acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

print("=== 5. Manual spot checks ===")
tests = [
    ("Win free iPhone click now", "spam"),
    ("Meeting at 10 AM tomorrow", "ham"),
    ("URGENT claim your lottery prize", "spam"),
    ("Can we reschedule lunch to Friday?", "ham"),
]
ok = 0
for text, expected in tests:
    pred, score = predict_message(pipe, text)
    match = pred == expected
    ok += int(match)
    mark = "OK" if match else "WRONG"
    sc = f"{score:.1%}" if score else "n/a"
    print(f"  [{mark}] expected={expected:4} got={pred:4} ({sc}) | {text[:45]}")
print(f"  Spot checks: {ok}/{len(tests)} passed")

print("\n=== 6. Batch sample file ===")
sample = Path("data/sample_inbox_scored.csv")
if sample.exists():
    print(pd.read_csv(sample).to_string(index=False))
else:
    print("  Run: python spam_classifier.py --batch data/sample_inbox.csv")

print("\n=== VERDICT ===")
if acc >= 0.95 and ok == len(tests):
    print("  Model looks CORRECT and healthy.")
elif acc >= 0.90:
    print("  Model works; metrics are acceptable.")
else:
    print("  Retrain: python spam_classifier.py --train")
