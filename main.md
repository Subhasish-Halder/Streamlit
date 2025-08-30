Twitter Sentiment — Training Script

What this script does?

Think of the pipeline like a tiny factory:

Load raw tweets → pick which columns are the “text” and the “label”.

Clean the text → remove punctuation/numbers, make lowercase.

Turn words into numbers → build a vocabulary (Tokenizer) and convert text to integer sequences.

Make all sequences same length → pad/truncate.

Map labels to integers → e.g., negative→0, neutral→1, positive→2.

Train an LSTM model → the model learns patterns.

Save everything → model (.keras), vocabulary (tokenizer_word_index.json), and config (config.json) for the app to use later.

Quick start

Put your Twitter_Data.csv in the same folder.

Run:

python twitter_sentiment_main.py


Outputs created:

sentiment_model.keras – the trained model (modern, version-safe)

tokenizer_word_index.json – word → index mapping

config.json – settings used during training (must match at inference)

Dataset expectations

Your CSV should have:

one text column (often text, Tweet, review, etc.)

one label column (often sentiment, category, label, etc.)

The script auto-detects these. If your file is unusual, you can hardcode names later (see “Use-cases” below).

Step-by-step walkthrough

1) Imports
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


What it does:

pandas reads CSVs, numpy for arrays.

train_test_split splits data into training/testing.

Tokenizer turns words into integers.

pad_sequences equalizes lengths.

Embedding, LSTM, Dense, Dropout are the model building blocks.

Analogy: tools in your workshop: saw, hammer, drill.

Use-cases:

If you move to transformers later, you’d replace the Keras Tokenizer with a HF tokenizer.

2) Config (tuneable knobs)
NUM_WORDS = 5000
MAX_LEN = 100
EPOCHS = 5
BATCH_SIZE = 32


What it does:

NUM_WORDS: keep top 5k most frequent words; others are ignored.

MAX_LEN: each tweet becomes a list of exactly 100 integers (after padding/truncation).

EPOCHS: how many passes over the data.

BATCH_SIZE: how many samples the model sees at once.

Analogy:

NUM_WORDS is like the size of your dictionary.

MAX_LEN is like fixing the length of a resume so they’re easy to compare.

Use-cases:

Short tweets? MAX_LEN=50 might be enough.

More data? You can raise NUM_WORDS to 10k–20k.

3) Candidate column names
TEXT_CANDIDATES = ["text", "Text", "Tweet", "tweet", "content", "review", "message"]
LABEL_CANDIDATES = ["sentiment", "Sentiment", "category", "Category", "label", "Label", "target", "polarity"]


What it does:
Lists common names so the script can auto-pick your text & label columns.

Use-cases:

If your CSV uses comment_body for text, add "comment_body" here or hardcode later.

4) Helper: pick which column to use
def pick_column(df, candidates, kind):
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
        others = [c for c in df.columns if c not in TEXT_CANDIDATES]
        if others:
            return others[-1]
    raise ValueError(...)


What it does:

Try known names first; otherwise guess a reasonable fallback (first text column for tweets, last non-text for labels).

Analogy:
Like scanning a spreadsheet for the most likely column names.

Use-cases:

If auto-detect picks wrong column, just replace with text_col = "my_text" / label_col = "my_label".

5) Helper: clean text
def clean_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[^a-zA-Z\s]", "", regex=True)
         .str.lower()
         .str.strip()
    )


What it does:

Keeps only letters/spaces, lowercases, trims spaces.

Example:
"Data-Science!!! 2025 😊" → "datascience"

Analogy:
Washing fruit before cooking.

Use-cases:

If you want to keep emojis or hashtags, remove or modify the regex.

Multi-lingual? Current regex removes non-ASCII characters; adapt accordingly.

6) Helper: map labels to ints
def map_labels(series: pd.Series):
    ...


What it does:

Converts labels (strings or numbers) to integers the model can learn:

"negative"/"neutral"/"positive" → 0/1/2

Numeric sets like {-1,0,1} or {0,2,4} are normalized to 0,1,2.

If only two classes, it reduces to binary (0/1).

Analogy:
Turning categories into jersey numbers.

Use-cases:

If your labels are custom like "bad"/"ok"/"great", extend the mapping.

7) Load CSV & pick columns
data = pd.read_csv("Twitter_Data.csv")
print("CSV columns:", data.columns.tolist())

text_col = pick_column(data, TEXT_CANDIDATES, kind="text")
label_col = pick_column(data, LABEL_CANDIDATES, kind="label")
print(f"Using text column  : {text_col}")
print(f"Using label column : {label_col}")


What it does:

Reads the CSV and tells you which columns it picked.

Great for sanity checking.

8) Clean text & align labels
clean_text = clean_text_series(data[text_col])
y_int, LABELS, valid_mask = map_labels(data[label_col])
X = clean_text.loc[valid_mask.index[valid_mask]]
y = y_int

if X.empty or y.empty:
    raise ValueError(...)


What it does:

Creates a cleaned text Series.

Maps labels to integers and drops unknowns.

Ensures X and y are aligned (same valid rows).

Analogy:
Line up students (texts) and their report cards (labels), removing incomplete pairs.

Use-cases:

If many rows drop, check your label values — you may need to broaden mappings.

9) Train/test split (with stratify if possible)
counts = y.value_counts()
can_stratify = (counts.min() >= 2) and (len(counts) >= 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
)


What it does:

Splits data into 80% train / 20% test.

Uses stratify if every class has ≥2 samples → keeps class ratios stable.

Analogy:
Ensuring each team has similar numbers of players.

Use-cases:

Tiny datasets sometimes can’t stratify; this guard avoids errors.

10) Tokenize → numbers, then pad/truncate
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="pre")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="pre")


What it does:

Learns a vocabulary from training text (most frequent words get low indices).

Converts each text to a list of word indices.

pad_sequences:

padding="post" → add zeros at the end if text is short.

truncating="pre" → cut from the start if text is too long.

Analogy:

Tokenizer = assigning an ID to every dictionary word.

Padding = making all lanes the same length for a fair race.

Important:

Words not in the top NUM_WORDS are ignored (become missing). This can lower confidence on very short or novel texts.

Optional improvement: use Tokenizer(num_words=..., oov_token="<OOV>") and store <OOV> index in your config so unseen words map to a special token instead of disappearing.

Use-cases:

If you have lots of rare/novel words, consider the OOV token.

11) One-hot encode labels
num_classes = len(sorted(y.unique()))
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes=num_classes)


What it does:

Converts label integers (e.g., 2) into vectors (e.g., [0,0,1]) for categorical_crossentropy.

Analogy:
Turning a jersey number into a scoreboard light.

Use-cases:

For binary classification you could also use a single sigmoid output and binary_crossentropy, but this script keeps a unified softmax path.

12) Build the LSTM model
model = tf.keras.Sequential([
    Embedding(input_dim=NUM_WORDS, output_dim=100, input_length=MAX_LEN),
    LSTM(128),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


What it does:

Embedding: learns a dense vector for each word (like giving words GPS coordinates).

LSTM(128): reads the sequence and remembers context (great for text).

Dense + ReLU + Dropout: a small head for classification; Dropout reduces overfitting.

Final Dense (softmax): outputs a probability for each class.

Use-cases:

More capacity? Try Bidirectional(LSTM(128)) or add another LSTM.

Faster/stronger? Move to DistilBERT later.

13) Train
model.fit(
    X_train_pad, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_test_oh),
    verbose=1
)


What it does:

Trains for EPOCHS, checking validation accuracy each epoch.

Use-cases:

Add EarlyStopping to stop when val accuracy stops improving:

cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
model.fit(..., callbacks=[cb])

14) Save artifacts for the app
model.save("sentiment_model.keras")  # modern Keras format (avoid .h5)
with open("tokenizer_word_index.json", "w") as f:
    json.dump(tokenizer.word_index, f)
with open("config.json", "w") as f:
    json.dump({"num_words": NUM_WORDS, "max_length": MAX_LEN, "labels": LABELS}, f, indent=2)


What it does:

Saves the model in Keras v3 format (.keras) → avoids legacy .h5 headaches.

Saves word_index as JSON → portable across versions (no pickling).

Saves config so the app mirrors training (same NUM_WORDS, MAX_LEN, label names).

Analogy:
Packing your trained chef, recipe book, and ingredient index for the restaurant front-desk.

Use-cases:

If you add an OOV token during training, also save its index in config.json (e.g., "oov_index": tokenizer.word_index.get("<OOV>")).

Practical examples & analogies

Text cleaning:
"AI 4ever!!! #Data" → regex removes non-letters: "AI ever Data" → lowercased ai ever data.

Tokenization (no OOV):
If your vocabulary has {"ai": 5, "data": 11} and the sentence is "ai rocks data",
"rocks" is unseen → dropped → sequence might be [5, 11].

Padding/Truncation:
With MAX_LEN=5:

Short [5,11] → pad: [5,11,0,0,0]

Long [7,8,9,10,11,12] → pre-truncate → keeps last 5: [8,9,10,11,12]

Label mapping (strings):
"positive" → 2; "negative" → 0; "neutral" → 1.

Common Use-cases / When to tweak knobs

Very short inputs look Neutral: add an OOV token:

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")


and store oov_index in config.json. Update the app to use it.

Longer texts: increase MAX_LEN to 200–256.

Large vocabulary: raise NUM_WORDS to 10k–20k.

Imbalanced classes: use class weights:

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = dict(enumerate(weights))
model.fit(..., class_weight=class_weight)


Overfitting: add EarlyStopping, raise Dropout, or reduce LSTM units.

Troubleshooting

KeyError: 'clean_text'
You don’t need a pre-existing clean_text. This script creates a cleaned Series itself.

CUDA/GPU messages
Safe to ignore on CPU; just informational.

Low confidence on rare words
Without OOV, unseen words are dropped. Add oov_token during training to fix.

App predictions don’t match training
Ensure the app uses the exact tokenizer_word_index.json, config.json, and same cleaning + pad_sequences args.

Reproducibility tips

Keep Python/TF versions pinned (you already did via requirements.txt).

Save random seeds if you need exact repeatability (not strictly necessary here).

Always ship sentiment_model.keras, tokenizer_word_index.json, config.json together.