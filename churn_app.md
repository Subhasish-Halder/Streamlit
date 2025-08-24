Streamlit app for Customer Churn Prediction — explanation

What this app does 
----------------------------------
You load a trained machine-learning model (a Random Forest saved to disk), ask the user for a few customer details
(tenure, internet service, contract, monthly and total charges), convert those inputs into the numeric format the model expects,
and then the model predicts whether the customer is likely to churn (leave) or stay. The app shows the result with a green
“stay” message or a red “churn” warning.

analogy
-------------------------
Think of the model as a trained “advisor” who has studied many past customers. The Streamlit page is a short questionnaire
you hand to the advisor. You tick boxes and fill numbers; the advisor looks at these answers and says “likely to stay” or
“likely to churn.” Because the advisor learned from historical examples, it spots patterns that are hard to see by eye.

Line-by-line walkthrough
------------------------
1) Imports and model loading
   - `import streamlit as st`, `import pandas as pd`: build the web UI and (optionally) handle data structures.
   - `from joblib import load`: to read a previously saved model file.
   - `from sklearn.preprocessing import LabelEncoder`: imported but not used in this snippet (safe to remove).
   - `model = load('random_forest_model.joblib')`: loads the trained Random Forest classifier from disk. This file must
     exist in the same folder (or provide a full path).

2) App title and input widgets
   - `st.title(...)` and `st.header(...)` create headings.
   - `st.number_input(...)` collects numeric values:
       • Tenure (months with the company)
       • Monthly Charges (bill per month)
       • Total Charges (money paid so far)
     Note: `number_input` returns numbers (integers here because `step` wasn’t specified; you could set `step=1.0` for floats).
   - `st.selectbox(...)` collects categorical choices:
       • InternetService: 'DSL', 'Fiber optic', or 'No'
       • Contract: 'Month-to-month', 'One year', 'Two year'

3) Mapping categories to numbers
   - The model needs numbers, not text. The code uses a simple dictionary:
       `label_mapping = {'DSL':0, 'Fiber optic':1, 'No':2, 'Month-to-month':0, 'One year':1, 'Two year':2}`
     Then it converts the selected strings to integers:
       `internet_service = label_mapping[internet_service]`
       `contract = label_mapping[contract]`

   Important: This mapping **must match exactly** how the model was trained. If the model was trained with different codes
   or with one-hot encoding (separate 0/1 columns), predictions can be wrong. Ideally, save and reuse the same preprocessing
   pipeline that was used during training (e.g., with `sklearn`’s `ColumnTransformer` + `OneHotEncoder`, then `joblib.dump` the whole pipeline).

4) Making a prediction
   - `model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])`
     The features are passed as a 2D list (shape = 1 row × 5 columns). The order of columns **must** match the training order.
     The model returns `[0]` or `[1]` (e.g., 0 = stay, 1 = churn).

5) Displaying the result
   - If `prediction[0] == 0`: show `st.success("This customer is likely to stay.")`
   - Else: show `st.error("This customer is likely to churn.")`

How to read the result
----------------------
- The app outputs a **class label** (stay/churn). It does not show the **probability** by default.
- A probability view is often more helpful in business (e.g., “This customer has a 78% chance to churn”).
  With many models you can do:
      `proba = model.predict_proba([[...]] )[0, 1]`
  and then display that number and choose a custom threshold (e.g., alert if `proba >= 0.35`).

Small example to build intuition
--------------------------------
- If a customer has tenure = 1 month, high monthly charges, and total charges near zero, the model may lean toward “churn”
  (new customers on expensive plans sometimes leave early).
- If another customer has tenure = 48 months, moderate monthly charges, and high total charges, it may lean toward “stay”.

Important correctness notes and best practices
----------------------------------------------
1) Consistent preprocessing:
   - The integer codes for categories must **match training**. If the Random Forest was trained with one-hot encodings
     (columns like InternetService_Fiber optic, Contract_Two year), this simple mapping is **not** equivalent.
   - Best practice: fit a preprocessing pipeline during training (e.g., OneHotEncoder for categories, any scaling if used),
     wrap it with the model in a single pipeline, and `joblib.dump` that pipeline. In the app, load the pipeline and call
     `pipeline.predict(...)` with raw strings; it will handle encoding consistently.

2) Feature order:
   - The order `[tenure, internet_service, contract, monthly_charges, total_charges]` must be the **same order** used in training.
     If you trained on a DataFrame with columns in a different order, rearrange here accordingly.

3) Data types:
   - Some models are sensitive to integer vs float. To be safe, convert to float:
       `import numpy as np`
       `row = np.array([[tenure, internet_service, contract, monthly_charges, total_charges]], dtype=float)`

4) Probabilities and thresholds:
   - Accuracy can look good while missing many churners (class imbalance). Consider showing probability and
     metrics like precision/recall/F1/AUC when evaluating the model offline.

5) UI niceties (optional):
   - Set default `total_charges` to `tenure * monthly_charges` as a starting hint.
   - Use help text in Streamlit (e.g., `st.number_input(..., help="...")`).
   - Validate inputs (e.g., if `total_charges < tenure * monthly_charges`, warn the user).

Why Random Forest fits this task
--------------------------------
- Works well with mixed features and non-linear relationships.
- Doesn’t require feature scaling.
- Provides feature importance for explainability (useful offline).
- Robust baseline for churn tasks.

Summary
-------
This Streamlit app loads a saved Random Forest, collects user inputs, numerically encodes categories, predicts churn,
and displays a simple message. The most critical technical requirement is **preprocessing consistency**: the encoding
(and column order) used in the app must be identical to what the model saw during training. For a production-ready app,
bundle preprocessing + model into a single saved pipeline and (optionally) display predicted probabilities alongside
the class label.