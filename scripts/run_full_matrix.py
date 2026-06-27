"""Run all 8 groups x 6 models x 2 datasets and save complete results."""
import sys, yaml, logging, warnings, time
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/EmailHeaderAnomalyDetection_Extended')

import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path('/home/claude/EmailHeaderAnomalyDetection_Extended')

with open(ROOT / 'config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['cv_folds'] = 5

from run_all import generate_synthetic_dataset
from src.experiments.run_experiments import _compute_importance_for_dataset, _run_single_experiment
from src.features.feature_groups import get_feature_subset

df_A = generate_synthetic_dataset(700, 700, 'A')
df_B = generate_synthetic_dataset(350, 350, 'B')
path_A = ROOT / 'data/processed/synthetic_A.csv'
path_B = ROOT / 'data/processed/synthetic_B.csv'
df_A.to_csv(path_A, index=False)
df_B.to_csv(path_B, index=False)
print(f"Data generated: A={df_A.shape}, B={df_B.shape}", flush=True)

all_rows = []
for ds_label, csv_path in [('A', path_A), ('B', path_B)]:
    df = pd.read_csv(csv_path)
    y = df['label']
    imp = _compute_importance_for_dataset(df, y, cfg, ds_label)
    print(f"Importance done for Dataset {ds_label}", flush=True)
    for group in ['G1','G2','G3','G4','G5','G6','G7','G8']:
        fc = get_feature_subset(df, group, imp)
        if not fc: continue
        X = df[fc]
        for model in ['RF','SVM','MLP','KNN','Stacking','OC_SVM']:
            try:
                t0 = time.perf_counter()
                r = _run_single_experiment(model, X, y, cfg, group, 5)
                r.update({'dataset': ds_label, 'n_features': len(fc)})
                all_rows.append(r)
                print(f"  {ds_label}/{group}/{model:10s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"  ERR {ds_label}/{group}/{model}: {e}", flush=True)

df_out = pd.DataFrame(all_rows)
(ROOT / 'results/feature_combinations').mkdir(parents=True, exist_ok=True)
df_out.to_csv(ROOT / 'results/feature_combinations/comparison.csv', index=False)
baseline = df_out[df_out.get('feature_group', pd.Series(dtype=str)) == 'G6'] if 'feature_group' in df_out.columns else pd.DataFrame()
(ROOT / 'results/original_baseline').mkdir(parents=True, exist_ok=True)
baseline.to_csv(ROOT / 'results/original_baseline/baseline.csv', index=False)
print(f"\nDONE: {len(df_out)} experiment rows saved.", flush=True)
print(df_out[['dataset','model_name','feature_group','accuracy','f1','auc']].to_string(index=False), flush=True)
