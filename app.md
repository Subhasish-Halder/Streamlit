Sentiment Analysis App 


What this app does 

It loads a trained sentiment model and vocabulary, cleans your input text, turns it into numbers the model understands, and returns Negative / Neutral / Positive with probabilities — all in a simple web UI.

Think of it like a translator + judge:

The translator converts your sentence into a numeric language the model understands.

The judge (the model) reads those numbers and decides which class fits best.

File prerequisites

The app expects these three files in the same folder:

sentiment_model.keras — the trained model (modern Keras format).

tokenizer_word_index.json — the vocabulary mapping (word -> integer index) used during training.

config.json — training settings the app must mirror (e.g., num_words, max_length, labels).

If any are missing, the app stops with a friendly error. That’s on purpose — predictions must follow the exact same setup as training.

Code Walkthrough (block by block)
1) Imports
import os
import re
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


What this does

os, json, re: work with files, JSON, and text cleaning.

numpy: fast math for arrays/probabilities.

streamlit: the web UI.

tensorflow + pad_sequences: load the model and make all sequences the same length.

Analogy: These are the tools on your workbench: a file opener, a dictionary, a sponge for cleaning, and a calculator.

Use cases / when to change

If you move beyond Keras to a different framework, imports will change.

If you add visualizations, you might import pandas/matplotlib.

2) Paths & existence check
MODEL_PATH = "sentiment_model.keras"
VOCAB_PATH = "tokenizer_word_index.json"
CONFIG_PATH = "config.json"

if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(CONFIG_PATH)):
    st.error("Missing one or more files: sentiment_model.keras, tokenizer_word_index.json, config.json")
    st.stop()


What this does

Defines where your artifacts live.

Stops early if something is missing, instead of crashing later with a cryptic error.

Analogy: Before cooking, you check if you have the recipe (config), the chef (model), and the pantry list (vocab). If not, you don’t start the oven.

Use cases

If you store files elsewhere, change these paths.

For Docker or cloud deployments, mount these files in the working directory or adjust to absolute paths.

3) Load config & vocabulary
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
NUM_WORDS = int(cfg.get("num_words", 5000))
MAX_LEN   = int(cfg.get("max_length", 100))
LABELS    = cfg.get("labels", ["Negative", "Neutral", "Positive"])

with open(VOCAB_PATH) as f:
    WORD_INDEX = json.load(f)  # word -> index (1-based integers)


What this does

Reads the same settings used during training:

NUM_WORDS: keep top N words from the vocabulary.

MAX_LEN: fix sequence length (pad or truncate).

LABELS: display names for classes in the correct order.

Loads WORD_INDEX (e.g., "market": 421, "data": 73).

Why it matters
Predictions must mirror training. If training used MAX_LEN=100 and top 5000 words, inference must too — otherwise predictions drift.

Use cases

If you train with an OOV token (e.g., <OOV>), you can also store and read oov_index here to handle unknown words robustly at inference.

4) Robust model loader (Keras 3)
def load_model_robust(path: str):
    # safe_mode=False relaxes checks and avoids some deserialization/name-scope bugs
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)

model = load_model_robust(MODEL_PATH)


What this does

Loads the Keras model using safe_mode=False which is helpful with Keras 3 when the saved graph contains metadata that can trigger name-scope errors during load.

Analogy: Opening a document in a “compatibility mode” so minor formatting quirks don’t stop you from reading it.

Use cases

If you never see deserialization issues, you can omit safe_mode=False, but keeping it reduces “it works on my machine” surprises.

5) Text preprocessing (must match training)
def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()
    return text


What this does

Cleans text exactly like the training script:

Removes everything except letters and spaces.

Lowercases.

Trims extra spaces.

Example
"Data science market? 2025!! #AI" → "data science market"

Analogy: Washing and chopping ingredients the same way every time so the recipe tastes identical.

Use cases

If you want emojis, hashtags, or numbers to matter, change this regex in both training and app.

6) Turn words into sequences (indices)
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


What this does

Reimplements the essential part of Keras’s Tokenizer without pickling:

Splits text into tokens.

Looks up each token in WORD_INDEX.

Keeps it only if index < NUM_WORDS (the top-N cutoff from training).

Drops unknown tokens (no OOV handling here).

Why not just use Tokenizer here?

The full Tokenizer object was trained earlier and serialized in a version-safe way as just the word_index JSON. That avoids cross-version pickling errors (e.g., keras.src.preprocessing).

Important note

Words not in WORD_INDEX (unseen during training) are dropped.
If you want to keep unknown words as a special id (e.g., <OOV>), train with Tokenizer(..., oov_token="<OOV>"), save oov_index in config.json, and modify this function to use it as a fallback.

Analogy: Translating a sentence using a dictionary. If a word isn’t in the dictionary, you either skip it (current behavior) or write “(unknown)” (OOV behavior).

7) Predict function
def predict_sentiment(text: str):
    clean = preprocess_text(text)
    seq = texts_to_sequences([clean], WORD_INDEX, num_words=NUM_WORDS)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="pre")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], float(probs[pred_idx]), probs


What this does

Clean the input with the same rules as training.

Convert to indices using the same vocabulary.

Pad/Truncate to MAX_LEN:

padding="post" → add zeros at the end of short sequences.

truncating="pre" → if too long, keep the last MAX_LEN tokens.

model.predict returns per-class probabilities.
Takes the class with the highest probability and returns its label & confidence.

Example

Input: "data science market?"

After cleaning: "data science market"

Suppose data→73, science→151, market→421, then sequence could be [73, 151, 421] → padded to 100 integers → model outputs something like [0.29, 0.40, 0.31] → Neutral.

Use cases

For longer inputs (articles), consider larger MAX_LEN or a transformer model.

If your dataset is imbalanced, add class weights during training; it affects these probabilities.

8) Streamlit UI
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


What this does

Simple UI:

A title, a text area, and a button.

On click, it calls predict_sentiment and shows:

The winning label and a confidence score.

All class probabilities (so you can see if it was a close call).

Analogy: A receptionist (Streamlit) takes your sentence and passes it to the specialist (model), then prints the result.

Use cases

Add input validation (min length).

Show an explanatory bar chart for probabilities.

Cache the model load with @st.cache_resource if you reload often.

End-to-End Example (tiny walkthrough)

Input typed:

"AI dominates the data science market!"


Preprocess → "ai dominates the data science market"

Tokenize → maybe [5, 912, 7, 73, 151, 421] (depends on your vocab)

Pad to 100 integers → [5, 912, 7, 73, 151, 421, 0, 0, ..., 0]

Predict → probabilities like [0.10, 0.25, 0.65] → “Positive (0.650)”

If a word is unseen (not in WORD_INDEX or with index ≥ NUM_WORDS), it’s dropped.

Common Tweaks & When to Use Them

Short queries feel Neutral

Root cause: many words dropped as unknown, or little context.

Fix (training): use an OOV token and store its index in config.json.

Fix (app): use oov_index when a word isn’t found (keep a placeholder id instead of dropping).

Longer texts get truncated

Raise MAX_LEN in both training and app.

Imbalanced dataset (e.g., many more neutrals)

Use class_weight during training to balance the learning.

Performance

Add @st.cache_resource on model loading so Streamlit doesn’t reload the model on every rerun:

@st.cache_resource
def _load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
model = _load_model()


Better accuracy

Try Bidirectional(LSTM(128)), more epochs with EarlyStopping, or switch to a small transformer (e.g., DistilBERT) for richer language understanding.

Troubleshooting

“Missing one or more files…”
Ensure sentiment_model.keras, tokenizer_word_index.json, and config.json are in the same directory as app.py.

“Predictions don’t match training”
Check that NUM_WORDS, MAX_LEN, LABELS, and the exact same cleaning are used in both places. If training used OOV but app doesn’t, align them.

“Low confidence” on very short inputs
That’s common. Add OOV handling and/or collect more training data. You can also raise NUM_WORDS at training time.