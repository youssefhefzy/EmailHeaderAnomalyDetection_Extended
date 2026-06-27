# save_model_real.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------- 1. Load the real processed Dataset A already have ----------
df = pd.read_csv("data/processed/datasetA_full.csv")# from your previous run
print("Loaded dataset with shape:", df.shape)
print("Available columns:", sorted(df.columns.tolist()))

# ---------- 2. Define the full G6 feature list (same as before) ----------
G6_FEATURES = [
    'received_domain_count', 'has_message_id', 'has_date', 'has_from',
    'has_to', 'has_subject', 'has_content_type', 'has_mime_version',
    'has_reply_to', 'has_in_reply_to', 'has_x_mailer', 'has_x_originating_ip',
    'subject_len', 'n_received_hops', 'tz_mismatch', 'received_time_variance',
    'n_distinct_ips', 'n_distinct_helos', 'from_domain_eq_reply_to',
    'from_domain_eq_return_path', 'from_domain_in_message_id',
    'has_duplicate_received', 'ip_private_in_received',
    'from_domain_in_received', 'helo_is_ip', 'is_base64_encoded',
    'malformed_date', 'mid_malformed', 'x_spam_score', 'has_x_priority'
]

# ---------- 3. Keep only features that are present in the CSV ----------
present = [f for f in G6_FEATURES if f in df.columns]
missing = [f for f in G6_FEATURES if f not in df.columns]
if missing:
    print("Missing G6 features in this dataset:", missing)
print("Using", len(present), "features for training.")

# ---------- 4. Prepare X and y ----------
X = df[present]
y = df['label']

# ---------- 5. Train Random Forest (same hyperparams as your experiments) ----------
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=1)
model.fit(X, y)
print(f"Training accuracy on full dataset: {model.score(X, y):.4f}")

# ---------- 6. Save ----------
out = {'model': model, 'features': present}
joblib.dump(out, 'header_model.joblib')
print(f"\nModel saved to header_model.joblib with {len(present)} features.")
print("Features:", present)