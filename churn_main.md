"""
Training and saving a Random Forest churn model — explanation

What this script does
-------------------------------------
1) Loads the Telco Customer Churn CSV into a pandas DataFrame.
2) Cleans and prepares a few columns so they’re numeric and model-friendly.
3) Encodes text categories (like InternetService and Contract) into numbers.
4) Chooses five input columns (features) and one target column (Churn).
5) Trains a Random Forest classifier to predict Churn.
6) Saves the trained model to disk (`random_forest_model.joblib`) so an app can load and use it later.

analogy
-------------------------
Imagine you’re building a “churn detector” robot:
- You show it lots of past customer records (training data).
- First you tidy the notes: fix messy numbers, convert words to codes the robot understands.
- Then the robot studies patterns (training).
- Finally, you freeze the robot’s brain in a file (joblib) so you can deploy it and ask for predictions later.

Line-by-line walkthrough
------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# 1) Load the raw data
telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

# 2) Clean 'TotalCharges' (make it numeric; fill blanks)
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)
- Many CSVs load numbers as text if there are blanks/spaces. `to_numeric(..., errors='coerce')` turns
  unparseable entries into NaN; `fillna(0)` replaces NaNs with 0.

# 3) Convert target 'Churn' to 0/1
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])
- For typical Telco data, 'No'→0, 'Yes'→1 (alphabetical order). You can check via `label_encoder.classes_`.

# 4) Label-encode two categorical inputs
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract']        = label_encoder.fit_transform(telecom_cust['Contract'])
- LabelEncoder converts each unique string to an integer (e.g., 'DSL'→0, 'Fiber optic'→1, 'No'→2).
- Trees can handle these integer codes reasonably well, but they do impose an ordering. One-hot encoding is often safer for linear models.

# 5) Select feature columns and target
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# 6) Train a Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X, y)
- A Random Forest is many decision trees voting together. It handles non-linear patterns and mixed feature types. No scaling needed.

# 7) Save the trained model
dump(model, 'random_forest_model.joblib')
- This writes the model to disk so another script/app (e.g., Streamlit) can load and predict.

Tiny examples to build intuition
--------------------------------
• LabelEncoder mappings (illustrative; actual mapping depends on sorted unique values):
  - InternetService classes: ['DSL', 'Fiber optic', 'No'] → codes [0, 1, 2]
    (Check with: LabelEncoder().fit(['DSL','Fiber optic','No']).classes_)
  - Contract classes: ['Month-to-month', 'One year', 'Two year'] → codes [0, 1, 2]
• If a customer has tenure=24, InternetService='Fiber optic'(1), Contract='Month-to-month'(0),
  MonthlyCharges=95, TotalCharges=2200 → the model uses these 5 numbers to decide 0 (stay) or 1 (churn).

Why these choices (and some cautions)
-------------------------------------
• RandomForest: Good baseline on tabular data, robust to outliers, handles interactions automatically, no scaling required.
• Label encoding vs one-hot:
  - Trees can work with integer codes, but the numeric order (0<1<2) is arbitrary for categories.
  - OneHotEncoder (0/1 columns per level) is often preferred for generality; for Random Forest, label encoding is usually acceptable.
• Filling TotalCharges with 0:
  - Simple and safe for code execution, but may not be statistically ideal. Alternatives:
    median/mean imputation, or a dedicated “missing” indicator.
• Training on the whole dataset:
  - This script doesn’t evaluate performance. In practice, do a train/test split (or CV), tune hyperparameters, and only then save the final model.

Best practices for deployment consistency
-----------------------------------------
1) Keep preprocessing identical at inference time:
   - The encodings learned here must be applied the same way in your app. A robust approach is to build a single
     `Pipeline` (preprocessing + model) and `dump` that pipeline, then `load` and call `predict` with raw strings.
2) Persist encoders if used separately:
   - If you rely on manual mappings in an app, ensure they match the training LabelEncoder’s alphabetical mapping.
   - To avoid drift, save the encoders too (e.g., `dump(le_internet, 'internet_encoder.joblib')`).

Optional improvements
---------------------
• Evaluate before saving:
  from sklearn.model_selection import train_test_split
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  model.fit(Xtr, ytr)
  from sklearn.metrics import classification_report, roc_auc_score
  yp = model.predict(Xte); yp_prob = model.predict_proba(Xte)[:,1]
  print(classification_report(yte, yp, digits=3))
  print("ROC AUC:", roc_auc_score(yte, yp_prob))
• Tune the forest:
  RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight='balanced', random_state=42)

Summary
-------
This script loads and cleans data, encodes a few categorical fields, selects five features, trains a Random Forest
to predict churn, and saves the model to disk. Think of it as building and packaging a trained “advisor” so you can
plug it into an app later. Just make sure the exact same preprocessing is applied when you make predictions in production.
"""
