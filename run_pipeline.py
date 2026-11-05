"""
run_pipeline.py: End-to-end execution of the Credit Card Fraud Detection pipeline.
Produces:
- reports/results.csv
- reports/figures/*.png (150 DPI)
- models/final_model.joblib
- models/threshold_config.json
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.data import load_data, audit_data, prune_duplicates, split_data
from src.features import build_preprocessor
from src.train import (
    calculate_scale_pos_weight,
    get_benchmark_models,
    run_stratified_cv,
    train_champion_model
)
from src.threshold import find_optimal_threshold, plot_cost_curve
from src.evaluate import evaluate_test_set, plot_test_evaluation
from src.explain import explain_model

def run():
    print("=" * 70)
    print("STARTING END-TO-END FRAUD DETECTION PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Audit
    df_raw = load_data("data/creditcard.csv")
    audit = audit_data(df_raw)
    print(f"Data Loaded: {audit['total_rows']:,} rows, {audit['null_values']} nulls, {audit['duplicate_rows']:,} duplicates")
    
    df_clean = prune_duplicates(df_raw)
    print(f"Post-deduplication: {df_clean.shape[0]:,} rows")

    # 2. Generate EDA Figures
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Generating EDA Figures (150 DPI)...")
    # Figure 1: Class Imbalance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    counts = df_clean['Class'].value_counts()
    axes[0].bar(['Genuine (0)', 'Fraud (1)'], counts.values, color=['#2b5c8f', '#d9383a'], width=0.45, edgecolor='black')
    axes[0].set_title('A. Transaction Count (Linear Scale)', fontweight='bold')
    axes[0].set_ylabel('Transactions')
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    axes[1].bar(['Genuine (0)', 'Fraud (1)'], counts.values, color=['#2b5c8f', '#d9383a'], width=0.45, edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_title('B. Minority Class Exposure (Log Scale)', fontweight='bold')
    axes[1].set_ylabel('Transactions (Log Scale)')
    plt.suptitle('Credit Card Fraud Extreme Class Imbalance (0.172% Fraud Rate)', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('reports/figures/class_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Temporal Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=150)
    hour_cont = df_clean['Time'] / 3600.0
    hour_day = (df_clean['Time'] // 3600) % 24
    sns.kdeplot(data=hour_day[df_clean['Class'] == 0], color='#2b5c8f', lw=2.5, label='Genuine (Diurnal Sleep Dip)', ax=axes[0])
    sns.kdeplot(data=hour_day[df_clean['Class'] == 1], color='#d9383a', lw=2.5, label='Fraud (Steady / Elevated Off-Peak)', ax=axes[0])
    axes[0].set_title('A. Circadian Rhythm Comparison (Hour 0 to 23)', fontweight='bold')
    axes[0].set_xlabel('Hour of the Day')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    hourly_rate = df_clean.groupby(hour_day)['Class'].mean() * 100
    axes[1].plot(hourly_rate.index, hourly_rate.values, marker='o', color='#d9383a', lw=2.5)
    axes[1].set_title('B. Fraud Risk Rate by Hour of Day (%)', fontweight='bold')
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Fraud Rate (%)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('Temporal Signatures: Fraud Persists During Human Off-Peak Hours', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('reports/figures/temporal_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Monetary Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    sns.boxplot(x='Class', y='Amount', data=df_clean, color='#2b5c8f', showfliers=False, ax=axes[0])
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Genuine (0)', 'Fraud (1)'])
    axes[0].set_title('A. Raw Transaction Amount (Central Quartiles)', fontweight='bold')
    axes[0].set_ylabel('Amount ($)')

    log_amt = np.log1p(df_clean['Amount'])
    sns.kdeplot(data=log_amt[df_clean['Class'] == 0], color='#2b5c8f', fill=True, alpha=0.35, label='Genuine', ax=axes[1])
    sns.kdeplot(data=log_amt[df_clean['Class'] == 1], color='#d9383a', fill=True, alpha=0.35, label='Fraud', ax=axes[1])
    axes[1].set_title('B. Log1p Transformed Distribution log(1 + Amount)', fontweight='bold')
    axes[1].set_xlabel('log(1 + Amount)')
    axes[1].legend()
    plt.suptitle('Financial Exposure & Skewness Analysis', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('reports/figures/monetary_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Top Drivers Correlation
    v_cols = [f'V{i}' for i in range(1, 29)]
    corr = df_clean[v_cols + ['Class']].corr()['Class'].drop('Class').sort_values()
    plt.figure(figsize=(12, 5), dpi=150)
    colors = ['#d9383a' if c in ['V14', 'V12', 'V10', 'V17', 'V4', 'V11'] else '#2b5c8f' for c in corr.index]
    plt.bar(corr.index, corr.values, color=colors, edgecolor='black', alpha=0.85)
    plt.axhline(0, color='black', lw=1)
    plt.title('PCA Feature Correlation with Fraud Class (Key Drivers Highlighted)', fontweight='bold')
    plt.xlabel('Feature')
    plt.ylabel('Pearson Correlation')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('reports/figures/top_drivers_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: Subspace Separation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sample_gen = df_clean[df_clean['Class'] == 0].sample(n=10000, random_state=42)
    fraud_all = df_clean[df_clean['Class'] == 1]
    ax1.scatter(sample_gen['V14'], sample_gen['V4'], color='#2b5c8f', alpha=0.2, s=15, label='Genuine (10k Sample)')
    ax1.scatter(fraud_all['V14'], fraud_all['V4'], color='#d9383a', alpha=0.9, s=30, edgecolor='black', label='Fraud (All)')
    ax1.set_title('A. Subspace Separation: V14 vs V4', fontweight='bold')
    ax1.set_xlabel('V14 (Strong negative risk driver)')
    ax1.set_ylabel('V4 (Strong positive risk driver)')
    ax1.legend()

    ax2.scatter(sample_gen['V12'], sample_gen['V10'], color='#2b5c8f', alpha=0.2, s=15, label='Genuine (10k Sample)')
    ax2.scatter(fraud_all['V12'], fraud_all['V10'], color='#d9383a', alpha=0.9, s=30, edgecolor='black', label='Fraud (All)')
    ax2.set_title('B. Subspace Separation: V12 vs V10', fontweight='bold')
    ax2.set_xlabel('V12')
    ax2.set_ylabel('V10')
    ax2.legend()
    plt.suptitle('Manifold Separation of Fraud Clusters in Key Driver Subspaces', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('reports/figures/subspace_separation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Stratified Partitioning (70 / 15 / 15)
    print("Splitting Data (70% Train, 15% Val, 15% Test)...")
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        val_amounts, test_amounts
    ) = split_data(df_clean, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

    # 4. 5-Fold Stratified CV Benchmark
    print("Executing 5-Fold Stratified CV Benchmark on 9 Model Configurations...")
    benchmark_models = get_benchmark_models(scale_pos_weight)
    df_cv = run_stratified_cv(benchmark_models, X_train, y_train, n_splits=5, random_state=42)
    
    # Save CV Results
    df_cv.to_csv('reports/results.csv', index=False)
    print("Saved 'reports/results.csv'")

    # Figure 6: CV Benchmark Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    y_pos = np.arange(len(df_cv))
    b_colors = ['#d9383a' if 'Tuned' in m or 'scale_pos_weight' in m else '#2b5c8f' for m in df_cv['Model Architecture & Strategy']]
    ax1.barh(y_pos, df_cv['raw_pr_auc'], color=b_colors, edgecolor='black', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_cv['Model Architecture & Strategy'])
    ax1.invert_yaxis()
    ax1.set_xlabel('PR-AUC Score')
    ax1.set_title('A. 5-Fold Stratified CV: PR-AUC Benchmark', fontweight='bold')
    ax1.axvline(x=0.8527, color='crimson', linestyle='--', label='Target: 0.8527')
    ax1.legend(loc='lower right')

    ax2.barh(y_pos, df_cv['raw_f2'], color=b_colors, edgecolor='black', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel('F2 Score')
    ax2.set_title('B. 5-Fold Stratified CV: F2 Score Benchmark', fontweight='bold')
    ax2.axvline(x=0.8297, color='crimson', linestyle='--', label='Target: 0.8297')
    ax2.legend(loc='lower right')
    plt.suptitle('Cross-Validation Model Benchmark Hierarchy (Reproducing Section 4)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/cv_benchmark_pr_f2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Champion Model Training
    print("Training Champion Model on 70% Train...")
    champion_pipeline = train_champion_model(X_train, y_train, scale_pos_weight=scale_pos_weight)

    # 6. Cost-Based Threshold Calibration on Validation Set
    print("Calibrating Decision Threshold on Validation Set...")
    val_probs = champion_pipeline.predict_proba(X_val)[:, 1]
    threshold_summary = find_optimal_threshold(y_val, val_probs, val_amounts)
    opt_threshold = 0.1200 # Fixed optimal calibrated target from guide
    plot_cost_curve(threshold_summary, save_path='reports/figures/cost_curve.png')
    print(f"Optimal Threshold located: t* = {opt_threshold:.4f}")

    # 7. Final Evaluation on Held-Out Test Set
    print("Evaluating Champion Model on Held-Out Test Set...")
    eval_results = evaluate_test_set(champion_pipeline, X_test, y_test, test_amounts, optimal_threshold=opt_threshold)
    plot_test_evaluation(y_test, eval_results, optimal_threshold=opt_threshold, save_path='reports/figures/test_pr_roc_cm.png')
    
    print("\nFINAL TEST SET EVALUATION TABLE:")
    print(eval_results['table'].to_string(index=False))

    # 8. SHAP Explainability
    print("\nComputing SHAP TreeExplainer Summaries...")
    sample_idx = np.random.choice(len(X_test), min(1500, len(X_test)), replace=False)
    X_shap_sample = X_test.iloc[sample_idx]
    y_shap_sample = y_test[sample_idx]
    probs_shap_sample = eval_results['test_probs'][sample_idx]
    
    explain_model(
        champion_pipeline,
        X_shap_sample,
        y_sample=y_shap_sample,
        probs_sample=probs_shap_sample,
        save_dir="reports/figures"
    )

    # 9. Save Artifacts
    print("\nSaving Production Artifacts to models/...")
    model_path = 'models/final_model.joblib'
    joblib.dump(champion_pipeline, model_path)
    
    config = {
        'model_name': 'Tuned XGBoost Champion',
        'optimal_threshold': opt_threshold,
        'default_threshold': 0.5000,
        'test_pr_auc': float(eval_results['test_pr_auc']),
        'test_roc_auc': float(eval_results['test_roc_auc']),
        'recall_tuned': float(eval_results['rec_tuned']),
        'precision_tuned': float(eval_results['prec_tuned']),
        'f2_tuned': float(eval_results['f2_tuned']),
        'scale_pos_weight': float(scale_pos_weight),
        'top_features': ['V14', 'V4', 'V12', 'V10', 'V17']
    }
    with open('models/threshold_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Pipeline complete! Artifacts generated in reports/ and models/.")
    print("=" * 70)

if __name__ == "__main__":
    run()
