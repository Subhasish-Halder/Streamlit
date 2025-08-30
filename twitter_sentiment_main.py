# twitter_sentiment_main.py
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Config ----------------
NUM_WORDS = 5000
MAX_LEN = 100
EPOCHS = 5
BATCH_SIZE = 32

# ---------------- Candidate column names ----------------
TEXT_CANDIDATES = [
    "text", "Text", "Tweet", "tweet", "content", "review", "message"
]
LABEL_CANDIDATES = [
    "sentiment", "Sentiment", "category", "Category", "label", "Label", "target", "polarity"
]

# ---------------- Helpers ----------------
def pick_column(df, candidates, kind):
    """Pick the first matching column name from candidates; otherwise pick a reasonable fallback."""
    for c in candidates:
        if c in df.columns:
            return c
    if kind == "text":
        obj_cols = [c for c in df.columns if df[c].dtype == "O"]
        if obj_cols:
            return obj_cols[0]
    else:  # label
        non_text = [c for c in df.columns if df[c].dtype != "O"]
        if non_text:
            return non_text[-1]
        # Fallback: any column not obviously text-ish
        others = [c for c in df.columns if c not in TEXT_CANDIDATES]
        if others:
            return others[-1]
    raise ValueError(f"Could not auto-detect a {kind} column. CSV columns: {df.columns.tolist()}")

def clean_text_series(s: pd.Series) -> pd.Series:
    """Mirror inference-time cleaning."""
    return (
        s.astype(str)
         .str.replace(r"[^a-zA-Z\s]", "", regex=True)
         .str.lower()
         .str.strip()
    )

def map_labels(series: pd.Series):
    """
    Map labels to ints and return (y_int, label_names).
    Handles:
      - strings: negative/neutral/positive (case-insensitive) and common aliases
      - ints: {-1,0,1} -> {0,1,2}, {0,2,4}->{0,1,2}, {0,1}->{0,1}
      - generic numeric fallbacks
    Drops rows with unknown labels.
    """
    if series.dtype == "O":
        s = series.astype(str).str.strip().str.lower()
        strict3 = {"negative": 0, "neutral": 1, "positive": 2}
        strict2 = {"negative": 0, "positive": 1}

        uniq = set(s.unique())

        if uniq <= set(strict3.keys()):
            y = s.map(strict3); labels = ["Negative", "Neutral", "Positive"]
        elif uniq <= set(strict2.keys()):
            y = s.map(strict2); labels = ["Negative", "Positive"]
        else:
            # heuristic mapping based on keywords
            def map_str(v):
                if "neg" in v:
                    return 0
                if "pos" in v:
                    return 2
                if "neu" in v:
                    return 1
                if v in {"0", "-1"}:
                    return 0
                if v in {"1"}:
                    return 1
                if v in {"2", "4"}:
                    return 2
                return np.nan
            y = s.map(map_str)
            vals = sorted(pd.Series(y.dropna().unique()).astype(int).tolist())
            if vals == [0, 2]:
                y = y.map({0: 0, 2: 1}); labels = ["Negative", "Positive"]
            elif vals == [0, 1, 2]:
                labels = ["Negative", "Neutral", "Positive"]
            else:
                # fallback to binary if unsure
                y = y.map(lambda z: 0 if z == 0 else (1 if z in {1,2} else np.nan))
                labels = ["Negative", "Positive"]
    else:
        s = pd.to_numeric(series, errors="coerce")
        uniq = sorted(pd.Series(s.dropna().unique()).astype(int).tolist())
        if set(uniq) == {-1, 0, 1}:
            y = s.map({-1: 0, 0: 1, 1: 2}); labels = ["Negative", "Neutral", "Positive"]
        elif set(uniq) == {0, 2, 4}:
            y = s.map({0: 0, 2: 1, 4: 2}); labels = ["Negative", "Neutral", "Positive"]
        elif set(uniq) == {0, 1, 2}:
            y = s; labels = ["Negative", "Neutral", "Positive"]
        elif set(uniq) == {0, 1}:
            y = s; labels = ["Negative", "Positive"]
        else:
            if len(uniq) == 2:
                remap = {uniq[0]: 0, uniq[1]: 1}; y = s.map(remap); labels = ["Negative", "Positive"]
            elif len(uniq) >= 3:
                remap = {uniq[0]: 0, uniq[1]: 1, uniq[-1]: 2}; y = s.map(remap); labels = ["Negative", "Neutral", "Positive"]
            else:
                raise ValueError(f"Unusable label set: {uniq}")
    mask = y.notna()
    return y[mask].astype(int), labels, mask

# ---------------- Load data ----------------
data = pd.read_csv("Twitter_Data.csv")
print("CSV columns:", data.columns.tolist())

text_col = pick_column(data, TEXT_CANDIDATES, kind="text")
label_col = pick_column(data, LABEL_CANDIDATES, kind="label")
print(f"Using text column  : {text_col}")
print(f"Using label column : {label_col}")

# Clean text into a local Series (do NOT assume data['clean_text'] exists)
clean_text = clean_text_series(data[text_col])

# Map labels
y_int, LABELS, valid_mask = map_labels(data[label_col])

# Align text and labels on valid rows only
X = clean_text.loc[valid_mask.index[valid_mask]]
y = y_int

if X.empty or y.empty:
    raise ValueError("After preprocessing, no valid samples remained. Check your CSV.")

# ---------------- Split ----------------
# Use stratify only when each class has at least 2 examples
counts = y.value_counts()
can_stratify = (counts.min() >= 2) and (len(counts) >= 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if can_stratify else None
)

# ---------------- Tokenize ----------------
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="pre")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="pre")

num_classes = len(sorted(y.unique()))
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes=num_classes)

# ---------------- Model ----------------
model = tf.keras.Sequential([
    Embedding(input_dim=NUM_WORDS, output_dim=100, input_length=MAX_LEN),
    LSTM(128),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------- Train ----------------
model.fit(
    X_train_pad, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_test_oh),
    verbose=1
)

# ---------------- Save artifacts ----------------
model.save("sentiment_model.keras")  # modern Keras format (avoid .h5)
with open("tokenizer_word_index.json", "w") as f:
    json.dump(tokenizer.word_index, f)
with open("config.json", "w") as f:
    json.dump(
        {"num_words": NUM_WORDS, "max_length": MAX_LEN, "labels": LABELS},
        f,
        indent=2
    )

print("Done. Saved: sentiment_model.keras, tokenizer_word_index.json, config.json")
