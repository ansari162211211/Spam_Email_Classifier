"""
Spam Email Classifier — train and predict spam vs ham using text content.

Uses the UCI SMS Spam Collection (bundled as data/emails.csv).
Compares Multinomial Naive Bayes and Linear SVM with TF-IDF features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).resolve().parent / "data" / "emails.csv"
MODEL_PATH = Path(__file__).resolve().parent / "spam_model.joblib"
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, emails, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python spam_classifier.py --prepare-data"
        )
    df = pd.read_csv(path)
    df["message"] = df["message"].astype(str).map(clean_text)
    df = df[df["message"].str.len() > 0]
    return df["message"].tolist(), df["label"].tolist()


def make_pipeline(model_name: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    if model_name == "nb":
        classifier = MultinomialNB(alpha=0.1)
    elif model_name == "svm":
        classifier = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_and_compare(
    messages: list[str],
    labels: list[str],
) -> tuple[Pipeline, str, dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_key = "nb"
    best_f1 = -1.0

    for name, display in [("nb", "Naive Bayes"), ("svm", "Linear SVM")]:
        pipe = make_pipeline(name)
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, pos_label="spam"),
        }
        results[display] = metrics
        print(f"\n--- {display} ---")
        print(f"  5-fold CV F1 (weighted): {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
        print(f"  Test accuracy: {metrics['accuracy']:.3f}")
        print(f"  Test F1 (spam): {metrics['f1']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = display
            best_key = name

    best_pipeline = make_pipeline(best_key)
    best_pipeline.fit(X_train, y_train)
    y_all_pred = best_pipeline.predict(X_test)
    best_pipeline.fit(messages, labels)
    print(f"\nBest model: {best_name}")
    print("\nClassification report (held-out test set):")
    print(classification_report(y_test, y_all_pred, target_names=["ham", "spam"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_all_pred, labels=["ham", "spam"]))

    flat = {f"{model}_{k}": v for model, m in results.items() for k, v in m.items()}
    return best_pipeline, best_name, flat


def load_model(path: Path = MODEL_PATH) -> tuple[Pipeline, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run: python spam_classifier.py --train"
        )
    saved = joblib.load(path)
    return saved["pipeline"], saved.get("model_name", "saved model")


def ensure_model(path: Path = MODEL_PATH) -> tuple[Pipeline, str]:
    """Load saved model or train and save if missing."""
    if path.exists():
        return load_model(path)
    messages, labels = load_dataset()
    pipeline, best_name, _ = train_and_compare(messages, labels)
    joblib.dump({"pipeline": pipeline, "model_name": best_name}, path)
    return pipeline, best_name


def predict_batch(
    pipeline: Pipeline,
    messages: list[str],
) -> pd.DataFrame:
    rows = []
    for msg in messages:
        label, spam_prob = predict_message(pipeline, str(msg))
        rows.append(
            {
                "prediction": label,
                "spam_score": spam_prob if spam_prob is not None else None,
            }
        )
    return pd.DataFrame(rows)


def score_csv(
    input_path: Path,
    output_path: Path,
    text_column: str = "message",
    pipeline: Pipeline | None = None,
) -> pd.DataFrame:
    pipe = pipeline if pipeline is not None else load_model()[0]
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Columns: {list(df.columns)}"
        )
    preds = predict_batch(pipe, df[text_column].astype(str).tolist())
    out = pd.concat([df, preds], axis=1)
    out.to_csv(output_path, index=False)
    return out


def predict_message(pipeline: Pipeline, message: str) -> tuple[str, float | None]:
    cleaned = clean_text(message)
    label = pipeline.predict([cleaned])[0]
    prob: float | None = None
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(pipeline.named_steps["tfidf"].transform([cleaned]))[0]
        classes = list(clf.classes_)
        prob = float(proba[classes.index("spam")])
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(
            pipeline.named_steps["tfidf"].transform([cleaned])
        )[0]
        prob = float(1 / (1 + pow(2.718281828, -score)))
    return label, prob


def prepare_data() -> None:
    """Download UCI SMS Spam Collection and write data/emails.csv."""
    import csv
    import urllib.request
    import zipfile

    data_dir = DATA_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "smsspam.zip"
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00228/smsspamcollection.zip"
    )
    print(f"Downloading dataset to {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("SMSSpamCollection").decode("utf-8").strip().splitlines()
    with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])
        for line in lines:
            label, msg = line.split("\t", 1)
            writer.writerow([label, msg])
    print(f"Saved {len(lines)} messages to {DATA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spam email / SMS classifier")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models, compare metrics, and save the best pipeline",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Download UCI SMS Spam dataset into data/emails.csv",
    )
    parser.add_argument(
        "--batch",
        metavar="INPUT.csv",
        help="Score a CSV file; writes predictions next to input ( *_scored.csv )",
    )
    parser.add_argument(
        "--text-column",
        default="message",
        help="Column name with email/SMS text for --batch (default: message)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for --batch (default: INPUT_scored.csv)",
    )
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
        return

    if args.batch:
        input_path = Path(args.batch)
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_name(f"{input_path.stem}_scored.csv")
        )
        pipeline, model_name = ensure_model()
        print(f"Using {model_name}")
        out = score_csv(input_path, output_path, args.text_column, pipeline)
        spam_count = (out["prediction"] == "spam").sum()
        print(f"Scored {len(out)} rows -> {output_path}")
        print(f"  Spam: {spam_count}  Ham: {len(out) - spam_count}")
        return

    if args.train:
        print("Loading dataset...")
        messages, labels = load_dataset()
        print(f"Samples: {len(messages)} (spam: {labels.count('spam')}, ham: {labels.count('ham')})")
        pipeline, best_name, _ = train_and_compare(messages, labels)
        joblib.dump({"pipeline": pipeline, "model_name": best_name}, MODEL_PATH)
        print(f"\nSaved model to {MODEL_PATH}")
        return

    pipeline, best_name = ensure_model()
    print(f"Loaded {best_name} from {MODEL_PATH}")

    print("\nEnter email/SMS text to classify (empty line to quit).")
    while True:
        try:
            msg = input("\nMessage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            break
        label, spam_prob = predict_message(pipeline, msg)
        if spam_prob is not None:
            print(f"Prediction: {label.upper()}  (spam score: {spam_prob:.1%})")
        else:
            print(f"Prediction: {label.upper()}")


if __name__ == "__main__":
    main()
