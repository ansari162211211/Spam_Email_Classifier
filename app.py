"""Streamlit UI for the spam email classifier."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from spam_classifier import ensure_model, predict_batch, predict_message

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("Spam Email Classifier")
st.caption("Naive Bayes & Linear SVM · TF-IDF · UCI SMS Spam dataset")


@st.cache_resource
def get_model():
    with st.spinner("Loading model (first time may take ~30 seconds)..."):
        return ensure_model()


try:
    pipeline, model_name = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Model: **{model_name}**")
st.sidebar.markdown(
    "Retrain from terminal:\n\n`python spam_classifier.py --train`"
)

tab_single, tab_batch = st.tabs(["Single message", "Batch CSV"])

with tab_single:
    text = st.text_area(
        "Paste email or SMS text",
        height=140,
        placeholder="e.g. Congratulations! You won a free prize...",
    )
    if st.button("Classify", type="primary", disabled=not text.strip()):
        label, spam_prob = predict_message(pipeline, text)
        if label == "spam":
            st.error(f"SPAM — {spam_prob:.1%} spam confidence" if spam_prob else "SPAM")
        else:
            st.success(
                f"NOT SPAM (ham) — {1 - spam_prob:.1%} likely legitimate"
                if spam_prob is not None
                else "NOT SPAM (ham)"
            )

with tab_batch:
    st.markdown(
        "Upload a CSV with a **`message`** column (or rename your text column to `message`)."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col = "message" if "message" in df.columns else df.columns[0]
        if col != "message":
            st.info(f"Using column **{col}** as message text.")
        if st.button("Score all rows", type="primary"):
            preds = predict_batch(pipeline, df[col].astype(str).tolist())
            result = pd.concat([df, preds], axis=1)
            st.dataframe(result, use_container_width=True)
            spam_n = (result["prediction"] == "spam").sum()
            st.metric("Spam detected", spam_n, delta=f"of {len(result)} messages")
            buf = io.StringIO()
            result.to_csv(buf, index=False)
            st.download_button(
                "Download scored CSV",
                buf.getvalue(),
                file_name="scored_messages.csv",
                mime="text/csv",
            )
