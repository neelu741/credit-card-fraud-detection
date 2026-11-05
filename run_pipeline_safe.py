"""
run_pipeline_safe.py: Sets sandbox-safe env vars before importing anything,
then runs the full pipeline. Use this instead of run_pipeline.py in sandboxed environments.
"""
import os
import sys

# Must be set BEFORE any joblib/sklearn/xgboost import
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Force stdout to be unbuffered
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

# Now safe to import everything
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_curve, auc, fbeta_score,
    precision_recall_curve, confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

print("=" * 70)
print("STARTING END-TO-END FRAUD DETECTION PIPELINE (SAFE MODE)")
print("=" * 70)

# ── 1. Load & Audit ──────────────────────────────────────────────────────────
df = pd.read_csv("data/creditcard.csv")
n_dup = int(df.duplicated().sum())
fraud_n = int(df['Class'].sum())
print(f"Data Loaded: {len(df):,} rows | {df.isnull().sum().sum()} nulls | {n_dup:,} duplicates | {fraud_n} frauds ({fraud_n/len(df)*100:.3f}%)")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Post-deduplication: {len(df):,} rows")

PALETTE = {0: '#2b5c8f', 1: '#d9383a'}
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 2. EDA Figures ───────────────────────────────────────────────────────────
print("Generating EDA Figures (150 DPI)...")

# Fig 1: Class imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
counts = df['Class'].value_counts().sort_index()
axes[0].bar(['Genuine (0)', 'Fraud (1)'], counts.values, color=['#2b5c8f','#d9383a'], width=0.45, edgecolor='black')
axes[0].set_title('A. Transaction Count (Linear Scale)', fontweight='bold')
axes[0].set_ylabel('Transactions')
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,p: f'{int(x):,}'))
axes[1].bar(['Genuine (0)', 'Fraud (1)'], counts.values, color=['#2b5c8f','#d9383a'], width=0.45, edgecolor='black')
axes[1].set_yscale('log')
axes[1].set_title('B. Minority Class Exposure (Log Scale)', fontweight='bold')
axes[1].set_ylabel('Transactions (Log Scale)')
plt.suptitle('Credit Card Fraud Extreme Class Imbalance (0.172% Fraud Rate)', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('reports/figures/class_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [1/5] class_imbalance.png saved")

# Fig 2: Temporal dynamics
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=150)
hour_day = (df['Time'] // 3600) % 24
sns.kdeplot(data=hour_day[df['Class']==0], color='#2b5c8f', lw=2.5, label='Genuine', ax=axes[0])
sns.kdeplot(data=hour_day[df['Class']==1], color='#d9383a', lw=2.5, label='Fraud', ax=axes[0])
axes[0].set_title('A. Circadian Rhythm Comparison', fontweight='bold')
axes[0].set_xlabel('Hour of the Day'); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)
hourly_rate = df.groupby(hour_day)['Class'].mean() * 100
axes[1].plot(hourly_rate.index, hourly_rate.values, marker='o', color='#d9383a', lw=2.5)
axes[1].set_title('B. Fraud Risk Rate by Hour (%)', fontweight='bold')
axes[1].set_xlabel('Hour of the Day'); axes[1].set_ylabel('Fraud Rate (%)'); axes[1].grid(True, linestyle='--', alpha=0.5)
plt.suptitle('Temporal Signatures: Fraud Persists During Off-Peak Hours', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('reports/figures/temporal_dynamics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [2/5] temporal_dynamics.png saved")

# Fig 3: Monetary distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
sns.boxplot(x='Class', y='Amount', data=df, color='#2b5c8f', showfliers=False, ax=axes[0])
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(['Genuine (0)','Fraud (1)'])
axes[0].set_title('A. Raw Transaction Amount (IQR)', fontweight='bold'); axes[0].set_ylabel('Amount ($)')
log_amt = np.log1p(df['Amount'])
sns.kdeplot(data=log_amt[df['Class']==0], color='#2b5c8f', fill=True, alpha=0.35, label='Genuine', ax=axes[1])
sns.kdeplot(data=log_amt[df['Class']==1], color='#d9383a', fill=True, alpha=0.35, label='Fraud', ax=axes[1])
axes[1].set_title('B. Log1p Transformed Distribution', fontweight='bold')
axes[1].set_xlabel('log(1 + Amount)'); axes[1].legend()
plt.suptitle('Financial Exposure & Skewness Analysis', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('reports/figures/monetary_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [3/5] monetary_distribution.png saved")

# Fig 4: Top driver correlations
v_cols = [f'V{i}' for i in range(1,29)]
corr = df[v_cols+['Class']].corr()['Class'].drop('Class').sort_values()
plt.figure(figsize=(12, 5), dpi=150)
colors = ['#d9383a' if c in ['V14','V12','V10','V17','V4','V11'] else '#2b5c8f' for c in corr.index]
plt.bar(corr.index, corr.values, color=colors, edgecolor='black', alpha=0.85)
plt.axhline(0, color='black', lw=1)
plt.title('PCA Feature Correlation with Fraud Class (Key Drivers Highlighted)', fontweight='bold')
plt.xlabel('Feature'); plt.ylabel('Pearson Correlation'); plt.xticks(rotation=45); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('reports/figures/top_drivers_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [4/5] top_drivers_correlation.png saved")

# Fig 5: Subspace separation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
gen_s = df[df['Class']==0].sample(n=10000, random_state=42)
fraud_all = df[df['Class']==1]
ax1.scatter(gen_s['V14'], gen_s['V4'], color='#2b5c8f', alpha=0.2, s=15, label='Genuine (10k)')
ax1.scatter(fraud_all['V14'], fraud_all['V4'], color='#d9383a', alpha=0.9, s=30, edgecolor='black', label='Fraud (All)')
ax1.set_title('A. Subspace: V14 vs V4', fontweight='bold'); ax1.set_xlabel('V14'); ax1.set_ylabel('V4'); ax1.legend()
ax2.scatter(gen_s['V12'], gen_s['V10'], color='#2b5c8f', alpha=0.2, s=15, label='Genuine (10k)')
ax2.scatter(fraud_all['V12'], fraud_all['V10'], color='#d9383a', alpha=0.9, s=30, edgecolor='black', label='Fraud (All)')
ax2.set_title('B. Subspace: V12 vs V10', fontweight='bold'); ax2.set_xlabel('V12'); ax2.set_ylabel('V10'); ax2.legend()
plt.suptitle('Manifold Separation of Fraud Clusters in Key Driver Subspaces', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('reports/figures/subspace_separation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [5/5] subspace_separation.png saved")

# ── 3. Split ─────────────────────────────────────────────────────────────────
print("Splitting Data (70% Train / 15% Val / 15% Test)...")
X = df.drop(columns=['Class'])
y = df['Class'].values
amounts = df['Amount'].values

X_train, X_temp, y_train, y_temp, _, amt_temp = train_test_split(
    X, y, amounts, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test, val_amounts, test_amounts = train_test_split(
    X_temp, y_temp, amt_temp, test_size=0.50, random_state=42, stratify=y_temp)

scale_pos_weight = float(np.sum(y_train==0) / max(np.sum(y_train==1), 1))
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── 4. Preprocessor ──────────────────────────────────────────────────────────
preprocessor = Pipeline([
    ('scaler', StandardScaler())  # scales Time & Amount; V-cols already unit-normal
])

def make_xgb_pipe(extra_sampler=None, **xgb_kwargs):
    xgb_kwargs.setdefault('n_jobs', 1)
    clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42, **xgb_kwargs)
    steps = [('scaler', StandardScaler()), ('clf', clf)]
    if extra_sampler:
        return ImbPipeline([('scaler', StandardScaler()), ('sampler', extra_sampler), ('clf', clf)])
    return Pipeline(steps)

def make_rf_pipe(extra_sampler=None, **rf_kwargs):
    rf_kwargs.setdefault('n_jobs', 1)
    clf = RandomForestClassifier(random_state=42, **rf_kwargs)
    if extra_sampler:
        return ImbPipeline([('scaler', StandardScaler()), ('sampler', extra_sampler), ('clf', clf)])
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])

# ── 5. 5-Fold Stratified CV Benchmark ────────────────────────────────────────
print("Executing 5-Fold Stratified CV (9 architectures)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    'Dummy (Majority Class)': Pipeline([('scaler', StandardScaler()), ('clf', DummyClassifier(strategy='most_frequent'))]),
    'Logistic Regression (Balanced)': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))]),
    'Random Forest (No Resampling)': make_rf_pipe(n_estimators=50, max_depth=10, class_weight='balanced'),
    'Random Forest + SMOTE': make_rf_pipe(SMOTE(sampling_strategy=0.1, random_state=42), n_estimators=50, max_depth=10),
    'Random Forest + RUS': make_rf_pipe(RandomUnderSampler(sampling_strategy=0.1, random_state=42), n_estimators=50, max_depth=10),
    'XGBoost (Unweighted / Default)': make_xgb_pipe(n_estimators=60, learning_rate=0.3, max_depth=6),
    'XGBoost (scale_pos_weight)': make_xgb_pipe(n_estimators=60, learning_rate=0.1, max_depth=5, scale_pos_weight=scale_pos_weight),
    'XGBoost + SMOTE': make_xgb_pipe(SMOTE(sampling_strategy=0.1, random_state=42), n_estimators=60, learning_rate=0.1, max_depth=5),
    'XGBoost (Tuned via RandomizedSearch)': make_xgb_pipe(n_estimators=80, learning_rate=0.08, max_depth=4, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight),
}

cv_results = []
for i, (name, pipe) in enumerate(MODELS.items(), 1):
    print(f"  [{i}/9] {name} ...", flush=True)
    pr_aucs, roc_aucs, f2s = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        Xtr, ytr = X_train.iloc[tr_idx], y_train[tr_idx]
        Xva, yva = X_train.iloc[va_idx], y_train[va_idx]
        pipe.fit(Xtr, ytr)
        probs = pipe.predict_proba(Xva)[:, 1]
        pr_aucs.append(average_precision_score(yva, probs))
        fpr, tpr, _ = roc_curve(yva, probs)
        roc_aucs.append(auc(fpr, tpr))
        f2s.append(fbeta_score(yva, (probs>=0.5).astype(int), beta=2, zero_division=0))
    cv_results.append({
        'Model Architecture & Strategy': name,
        'PR-AUC (mean +/- std)': f"{np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}",
        'ROC-AUC (mean +/- std)': f"{np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}",
        'F2 Score (mean +/- std)': f"{np.mean(f2s):.4f} +/- {np.std(f2s):.4f}",
        'raw_pr_auc': np.mean(pr_aucs),
        'raw_f2': np.mean(f2s),
    })
    print(f"       PR-AUC={np.mean(pr_aucs):.4f}  ROC-AUC={np.mean(roc_aucs):.4f}  F2={np.mean(f2s):.4f}", flush=True)

df_cv = pd.DataFrame(cv_results)
df_cv.to_csv('reports/results.csv', index=False)
print("Saved reports/results.csv")

# Fig 6: CV Benchmark bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
y_pos = np.arange(len(df_cv))
b_col = ['#d9383a' if 'Tuned' in m or 'scale_pos_weight' in m else '#2b5c8f' for m in df_cv['Model Architecture & Strategy']]
ax1.barh(y_pos, df_cv['raw_pr_auc'], color=b_col, edgecolor='black', alpha=0.85)
ax1.set_yticks(y_pos); ax1.set_yticklabels(df_cv['Model Architecture & Strategy'])
ax1.invert_yaxis(); ax1.set_xlabel('PR-AUC'); ax1.set_title('A. 5-Fold CV: PR-AUC', fontweight='bold')
ax1.axvline(x=0.8527, color='crimson', linestyle='--', label='Target 0.8527'); ax1.legend(loc='lower right')
ax2.barh(y_pos, df_cv['raw_f2'], color=b_col, edgecolor='black', alpha=0.85)
ax2.set_yticks(y_pos); ax2.set_yticklabels([]); ax2.invert_yaxis()
ax2.set_xlabel('F2 Score'); ax2.set_title('B. 5-Fold CV: F2 Score', fontweight='bold')
ax2.axvline(x=0.8297, color='crimson', linestyle='--', label='Target 0.8297'); ax2.legend(loc='lower right')
plt.suptitle('Cross-Validation Model Benchmark Hierarchy', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/cv_benchmark_pr_f2.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 6. Champion Model ─────────────────────────────────────────────────────────
print("Training Champion XGBoost on full 70% Train partition...")
champion = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        n_estimators=80, learning_rate=0.08, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', random_state=42, n_jobs=1
    ))
])
champion.fit(X_train, y_train)
print("Champion model trained.")

# ── 7. Cost-Based Threshold Calibration ───────────────────────────────────────
print("Calibrating threshold on Validation set...")
val_probs = champion.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
costs = []
FP_COST = 10      # False positive: manual review cost ($)
FN_COST = 400     # False negative: average fraud loss ($)

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    fp = int(np.sum((preds==1) & (y_val==0)))
    fn = int(np.sum((preds==0) & (y_val==1)))
    costs.append(fp * FP_COST + fn * FN_COST)

opt_threshold = float(thresholds[np.argmin(costs)])
print(f"Optimal threshold t* = {opt_threshold:.4f} (guide target: 0.1200)")

# Fig 7: Cost curve
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(thresholds, costs, color='#2b5c8f', lw=2.5)
plt.axvline(opt_threshold, color='#d9383a', linestyle='--', lw=2, label=f't* = {opt_threshold:.4f}')
plt.xlabel('Decision Threshold'); plt.ylabel('Total Banking Cost ($)')
plt.title('Cost-Based Threshold Optimisation on Validation Set', fontweight='bold')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('reports/figures/cost_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 8. Final Test-Set Evaluation ──────────────────────────────────────────────
print("Evaluating on held-out Test Set...")
test_probs = champion.predict_proba(X_test)[:, 1]
test_preds_default = (test_probs >= 0.50).astype(int)
test_preds_tuned   = (test_probs >= opt_threshold).astype(int)

prec_d = precision_score(y_test, test_preds_default, zero_division=0)
rec_d  = recall_score(y_test, test_preds_default, zero_division=0)
f2_d   = fbeta_score(y_test, test_preds_default, beta=2, zero_division=0)

prec_t = precision_score(y_test, test_preds_tuned, zero_division=0)
rec_t  = recall_score(y_test, test_preds_tuned, zero_division=0)
f2_t   = fbeta_score(y_test, test_preds_tuned, beta=2, zero_division=0)

prec_arr, rec_arr, _ = precision_recall_curve(y_test, test_probs)
test_pr_auc = auc(rec_arr, prec_arr)
fpr_t, tpr_t, _ = roc_curve(y_test, test_probs)
test_roc_auc = auc(fpr_t, tpr_t)

print(f"\nFINAL TEST METRICS:")
print(f"  PR-AUC={test_pr_auc:.4f} | ROC-AUC={test_roc_auc:.4f}")
print(f"  Default  (t=0.50): Prec={prec_d:.4f}  Rec={rec_d:.4f}  F2={f2_d:.4f}")
print(f"  Tuned    (t={opt_threshold:.4f}): Prec={prec_t:.4f}  Rec={rec_t:.4f}  F2={f2_t:.4f}")

# Fig 8: PR curve, ROC curve, Confusion Matrix (2x2 panel)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
axes[0].plot(rec_arr, prec_arr, color='#2b5c8f', lw=2.5, label=f'PR-AUC={test_pr_auc:.4f}')
axes[0].axhline(y=y_test.mean(), color='gray', linestyle='--', label='Baseline')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('A. Precision-Recall Curve', fontweight='bold'); axes[0].legend()

axes[1].plot(fpr_t, tpr_t, color='#d9383a', lw=2.5, label=f'ROC-AUC={test_roc_auc:.4f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('B. ROC Curve', fontweight='bold'); axes[1].legend()

cm = confusion_matrix(y_test, test_preds_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Pred Genuine','Pred Fraud'], yticklabels=['Actual Genuine','Actual Fraud'])
axes[2].set_title(f'C. Confusion Matrix (t={opt_threshold:.4f})', fontweight='bold')

plt.suptitle('Final Test-Set Evaluation: Tuned XGBoost Champion', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/test_pr_roc_cm.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 9. SHAP Explanations ──────────────────────────────────────────────────────
print("Computing SHAP explanations...")
try:
    import shap
    xgb_model = champion.named_steps['clf']
    scaler = champion.named_steps['scaler']

    sample_idx = np.random.RandomState(42).choice(len(X_test), min(1500, len(X_test)), replace=False)
    X_shap = X_test.iloc[sample_idx]
    X_shap_scaled = pd.DataFrame(scaler.transform(X_shap), columns=X_shap.columns)

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_shap_scaled)

    # Global beeswarm
    plt.figure(figsize=(10, 8), dpi=150)
    shap.summary_plot(shap_vals, X_shap_scaled, show=False, max_display=15)
    plt.title('SHAP Global Feature Importance (Beeswarm)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Local waterfall for top fraud
    fraud_idx_in_sample = np.where(y_test[sample_idx] == 1)[0]
    if len(fraud_idx_in_sample) > 0:
        fi = fraud_idx_in_sample[0]
        shap_exp = shap.Explanation(
            values=shap_vals[fi],
            base_values=explainer.expected_value,
            data=X_shap_scaled.iloc[fi].values,
            feature_names=list(X_shap_scaled.columns)
        )
        plt.figure(figsize=(10, 6), dpi=150)
        shap.waterfall_plot(shap_exp, show=False, max_display=12)
        plt.title('SHAP Local Explanation: Fraud Case', fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/shap_waterfall_fraud.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("SHAP figures saved.")
except Exception as e:
    print(f"SHAP explanation skipped: {e}")

# ── 10. Save Artifacts ────────────────────────────────────────────────────────
print("Saving model artifacts...")
joblib.dump(champion, 'models/final_model.joblib')

config = {
    'model_name': 'Tuned XGBoost Champion',
    'optimal_threshold': round(opt_threshold, 4),
    'default_threshold': 0.5000,
    'test_pr_auc': round(test_pr_auc, 4),
    'test_roc_auc': round(test_roc_auc, 4),
    'recall_tuned': round(rec_t, 4),
    'precision_tuned': round(prec_t, 4),
    'f2_tuned': round(f2_t, 4),
    'scale_pos_weight': round(scale_pos_weight, 2),
    'top_features': ['V14', 'V4', 'V12', 'V10', 'V17']
}
with open('models/threshold_config.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"\nSaved: models/final_model.joblib")
print(f"Saved: models/threshold_config.json")
print(f"\nFigures in reports/figures/:")
for fn in sorted(os.listdir('reports/figures')):
    print(f"  {fn}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
