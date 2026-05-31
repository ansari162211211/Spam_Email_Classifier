# Spam Email Classifier

Machine learning project that classifies messages as **spam** or **ham** (legitimate) from text content. Built with scikit-learn for internship / portfolio use.

## Features

- **Real dataset** — [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (~5,500 labeled messages)
- **Text preprocessing** — normalization, URL/email removal
- **TF-IDF** — unigrams + bigrams, English stop words
- **Model comparison** — Multinomial Naive Bayes vs Linear SVM
- **Evaluation** — train/test split, 5-fold cross-validation, precision/recall/F1, confusion matrix
- **CLI** — train, interactive predict, batch CSV scoring
- **Web UI** — Streamlit app for demos

Typical test accuracy is **~98%** on held-out data.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

python spam_classifier.py --prepare-data
python spam_classifier.py --train
python spam_classifier.py
streamlit run app.py
```

## Usage

### Train

```bash
python spam_classifier.py --train
```

### Classify one message (CLI)

```bash
python spam_classifier.py
```

### Batch score a CSV

```bash
python spam_classifier.py --batch data/sample_inbox.csv
python spam_classifier.py --batch data.csv --text-column body -o results.csv
```

### Verify model

```bash
python verify_model.py
```

### Web app

```bash
streamlit run app.py
```

## Project structure

```
Spam_Email_Classifier/
├── spam_classifier.py
├── app.py
├── verify_model.py
├── data/emails.csv
├── spam_model.joblib
├── requirements.txt
└── README.md
```
