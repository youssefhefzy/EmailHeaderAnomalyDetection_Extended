# save_model_real.py - Train model with G6 + G5 features (24 total)
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---------- 1. Define the combined G6 + G5 feature list ----------
FEATURES_G6_G5 = [
    # G6 features (16 that exist in real data)
    'has_message_id', 'has_date', 'has_from', 'has_to', 'has_subject',
    'has_content_type', 'has_mime_version', 'has_reply_to', 'has_x_mailer',
    'has_x_originating_ip', 'subject_len', 'n_received_hops',
    'is_base64_encoded', 'malformed_date', 'mid_malformed', 'has_x_priority',
    # G5 authentication features (8)
    'spf_result', 'dkim_result', 'dmarc_result',
    'dkim_signature_present', 'dkim_signature_valid_format',
    'auth_results_present', 'all_auth_pass', 'any_auth_fail'
]

# ---------- 2. Load the real processed Dataset A ----------
try:
    df = pd.read_csv("data/processed/datasetA_full.csv")
    print("Loaded dataset with shape:", df.shape)
except FileNotFoundError:
    print("Dataset not found. Creating synthetic training data...")
    np.random.seed(42)
    rows = []
    for i in range(2000):
        rows.append({
            'has_message_id':1,'has_date':1,'has_from':1,'has_to':1,'has_subject':1,
            'has_content_type':1,'has_mime_version':1,'has_reply_to':np.random.choice([0,1]),
            'has_x_mailer':np.random.choice([0,1]),'has_x_originating_ip':0,
            'subject_len':int(np.random.normal(30,15)),
            'n_received_hops':int(np.random.choice([2,3,4,5,6])),
            'is_base64_encoded':0,'malformed_date':0,'mid_malformed':0,'has_x_priority':0,
            'spf_result':np.random.choice([1,0,-1],p=[0.6,0.3,0.1]),
            'dkim_result':np.random.choice([1,0,-1],p=[0.6,0.3,0.1]),
            'dmarc_result':np.random.choice([1,0,-1],p=[0.6,0.3,0.1]),
            'dkim_signature_present':np.random.choice([0,1],p=[0.3,0.7]),
            'dkim_signature_valid_format':np.random.choice([0,1],p=[0.3,0.7]),
            'auth_results_present':np.random.choice([0,1],p=[0.3,0.7]),
            'all_auth_pass':np.random.choice([0,1],p=[0.4,0.6]),
            'any_auth_fail':np.random.choice([0,1],p=[0.7,0.3]),
            'label':np.random.choice([0,1])
        })
    df = pd.DataFrame(rows)
    print("Created synthetic dataset with shape:", df.shape)

# ---------- 3. Keep only features that are present ----------
present = [f for f in FEATURES_G6_G5 if f in df.columns]
missing = [f for f in FEATURES_G6_G5 if f not in df.columns]
if missing:
    print("Missing features:", missing)
print("Using", len(present), "features for training (G6 + G5)")
print("Features:", present)

# ---------- 4. Prepare X and y ----------
X = df[present]
y = df['label']

# ---------- 5. Train Random Forest ----------
model = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_leaf=3,
    random_state=42, n_jobs=1
)
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.4f}")

# ---------- 6. Save ----------
out = {'model': model, 'features': present}
joblib.dump(out, 'header_model.joblib')
print(f"\nModel saved to header_model.joblib with {len(present)} features (G6 + G5)")
'@ | Out-File -FilePath "save_model_real.py" -Encoding UTF8'