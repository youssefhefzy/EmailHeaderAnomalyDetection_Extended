import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. LOAD PROCESSED DATA ----------
# Use your real Dataset A (change path if needed)
df = pd.read_csv('data/processed/datasetA_processed.csv')
label_col = 'label'
features = df.drop(columns=[label_col])

# ---------- 2. DEFINE FEATURE GROUPS (same as feature_groups.py) ----------
groups = {
    'G1': [c for c in features.columns if c.startswith('has_') and c not in
           ['has_dkim_signature', 'has_domainkey_signature', 'has_authentication_results',
            'has_arc_authentication_results', 'has_envelope_to', 'has_spf', 'has_dkim',
            'has_dmarc']],  # header presence flags (excluding auth ones handled in G5)
    'G2': ['from_domain_eq_reply_to', 'from_domain_eq_return_path', 'from_domain_eq_msg_id',
           'reply_to_domain_eq_msg_id', 'return_path_domain_eq_msg_id',
           'from_domain_eq_received_domain', 'reply_to_subdomain_of_from',
           'from_domain_in_message_id', 'received_domain_count', 'unique_received_domains',
           'received_domain_variance'],
    'G3': ['n_received_hops', 'received_time_variance', 'tz_mismatch',
           'n_distinct_ips', 'n_distinct_helos', 'n_distinct_froms',
           'has_duplicate_received', 'received_chain_anomaly',
           'ip_private_in_received', 'helo_is_ip', 'from_domain_in_received'],
    'G4': ['subject_len', 'subject_non_ascii', 'from_non_ascii', 'malformed_date',
           'is_base64_encoded', 'mid_malformed', 'x_priority_value',
           'has_x_mailer', 'has_x_originating_ip', 'has_reply_to',
           'has_in_reply_to', 'has_content_type', 'has_mime_version',
           'has_x_mime_ole', 'has_x_priority', 'n_bcc_addresses'],
    'G5': ['spf_result', 'dkim_result', 'dmarc_result', 'dkim_signature_present',
           'dkim_signature_valid_format', 'auth_results_present',
           'all_auth_pass', 'any_auth_fail'],
    'G6': [],  # Top-30 by importance, will be filled from config or list
    'G7': [],  # G2+G3+G5
    'G8': [],  # Top-5 fastest inference
}

# G6 is the top-30 from paper (we can get from config or hardcode)
# For simplicity, we use a known set of headers from the paper's G6
# (These are usually the features with highest permutation importance)
g6_features = [
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
# Filter only those present in our DataFrame
g6_features = [f for f in g6_features if f in features.columns]
groups['G6'] = g6_features
groups['G7'] = groups['G2'] + groups['G3'] + groups['G5']  # union
groups['G8'] = g6_features[:5] if len(g6_features) >=5 else g6_features

# Remove any group that is empty (e.g., G5 might be missing on Dataset A)
groups = {k: v for k, v in groups.items() if v}

# Flatten all features that have a group assignment
all_grouped = []
for lst in groups.values():
    all_grouped.extend(lst)
all_grouped = list(dict.fromkeys(all_grouped))  # remove duplicates

# Keep only features that exist in the DataFrame
all_grouped = [f for f in all_grouped if f in features.columns]
X = features[all_grouped].copy()

# ---------- 3. CORRELATION MATRIX ----------
# Spearman rank correlation handles mixed types and non-linear monotonic relationships
corr = X.corr(method='spearman')

# ---------- 4. CLUSTERED HEATMAP WITH GROUP COLORS ----------
# Compute linkage for clustering
dist = 1 - corr.abs()   # distance based on absolute correlation
condensed_dist = squareform(dist, checks=False)
linkage = hierarchy.ward(condensed_dist)
order = hierarchy.leaves_list(linkage)
corr_clustered = corr.iloc[order, order]

# Create a color map for groups
group_colors = {}
color_palette = sns.color_palette('tab10', len(groups))
for idx, (gname, feats) in enumerate(groups.items()):
    for f in feats:
        if f in X.columns:
            group_colors[f] = color_palette[idx]

# Build a row/col color vector in the clustered order
row_colors = [group_colors[f] for f in corr_clustered.columns]

# Plot
plt.figure(figsize=(20, 16))
g = sns.clustermap(corr_clustered, cmap='coolwarm', center=0,
                   row_cluster=False, col_cluster=False,  # already clustered
                   row_colors=row_colors, col_colors=row_colors,
                   xticklabels=True, yticklabels=True,
                   linewidths=0, cbar_kws={'label': 'Spearman correlation'},
                   dendrogram_ratio=0.1)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)

# Add legend for groups
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=color_palette[i], label=gname) for i, gname in enumerate(groups.keys())]
plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.05, 1))
plt.suptitle('Clustered Correlation Matrix of Email Header Features by Group', fontsize=14)
g.savefig('results/figures/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()  # or close if non-interactive

# ---------- 5. WITHIN-GROUP vs BETWEEN-GROUP CORRELATION ----------
print("\n" + "="*70)
print("AVERAGE ABSOLUTE CORRELATION WITHIN EACH GROUP")
print("="*70)
within_corrs = {}
for gname, feats in groups.items():
    feats_present = [f for f in feats if f in X.columns]
    if len(feats_present) < 2:
        within_corrs[gname] = np.nan
        continue
    sub_corr = corr.loc[feats_present, feats_present]
    # Take upper triangle (excluding diagonal)
    triu = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
    avg = triu.abs().mean().mean()
    within_corrs[gname] = avg
    print(f"{gname:6s}: {avg:.3f}  (n={len(feats_present)} features)")

# Overall between-group: average absolute correlation of features belonging to different groups
all_features_in_groups = [f for lst in groups.values() for f in lst if f in X.columns]
between_sum = 0.0
between_count = 0
for i, f1 in enumerate(all_features_in_groups):
    for j, f2 in enumerate(all_features_in_groups):
        if i >= j:
            continue
        # Check if they belong to different groups
        g1 = next((g for g, feats in groups.items() if f1 in feats), None)
        g2 = next((g for g, feats in groups.items() if f2 in feats), None)
        if g1 != g2:
            between_sum += abs(corr.loc[f1, f2])
            between_count += 1
avg_between = between_sum / between_count if between_count > 0 else 0

print(f"\nAverage ABSOLUTE correlation BETWEEN different groups: {avg_between:.3f}")
print(f"Average ABSOLUTE correlation WITHIN groups (mean of above): {np.nanmean(list(within_corrs.values())):.3f}")
print(f"Ratio within/between: {np.nanmean(list(within_corrs.values()))/avg_between:.2f}")