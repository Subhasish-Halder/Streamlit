import os
import re
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------- Paths (ensure these files exist) --------
MODEL_PATH = "sentiment_model.keras"
VOCAB_PATH = "tokenizer_word_index.json"
CONFIG_PATH = "config.json"

# -------- Load config & vocab --------
if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(CONFIG_PATH)):
    st.error("Missing one or more files: sentiment_model.keras, tokenizer_word_index.json, config.json")
    st.stop()

with open(CONFIG_PATH) as f:
    cfg = json.load(f)
NUM_WORDS = int(cfg.get("num_words", 5000))
MAX_LEN   = int(cfg.get("max_length", 100))
LABELS    = cfg.get("labels", ["Negative", "Neutral", "Positive"])

with open(VOCAB_PATH) as f:
    WORD_INDEX = json.load(f)  # word -> index (1-based integers)

# -------- Robust model loader (Keras 3) --------
def load_model_robust(path: str):
    # safe_mode=False relaxes checks and avoids some deserialization/name-scope bugs
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

model = load_model_robust(MODEL_PATH)

# -------- Preprocessing (must match training) --------
def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()
    return text

def texts_to_sequences(texts, word_index, num_words=None):
    seqs = []
    for t in texts:
        tokens = t.split()
        indices = []
        for tok in tokens:
            idx = word_index.get(tok)
            if idx is not None:
                idx = int(idx)
                if (num_words is None) or (idx < num_words):
                    indices.append(idx)
        seqs.append(indices)
    return seqs

def predict_sentiment(text: str):
    clean = preprocess_text(text)
    seq = texts_to_sequences([clean], WORD_INDEX, num_words=NUM_WORDS)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="pre")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], float(probs[pred_idx]), probs

# -------- Streamlit UI --------
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    text = user_input.strip()
    if not text:
        st.warning("Please enter some text for analysis.")
    else:
        label, conf, probs = predict_sentiment(text)
        st.markdown(f"**Prediction:** {label}  \n**Confidence:** {conf:.3f}")
        st.subheader("Class probabilities")
        for i, p in enumerate(probs):
            st.write(f"{LABELS[i]}: {p:.3f}")
