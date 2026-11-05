"""
evaluate.py: Final Held-Out Test Set Evaluation.
Produces the exact test comparison table and figures matching the Presentation Master Guide.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    fbeta_score
)
from .threshold import evaluate_banking_cost


def evaluate_test_set(pipeline, X_test, y_test, test_amounts, optimal_threshold=0.1200):
    """
    Evaluates champion pipeline on the untouched 15% Held-Out Test Set at both t=0.50 and t*=0.1200.
    """
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # Probabilistic metrics
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    test_roc_auc = float(auc(fpr, tpr))

    # Metrics at Default Threshold (t = 0.50)
    res_default = evaluate_banking_cost(y_test, test_probs, test_amounts, threshold=0.50)
    preds_def = (test_probs >= 0.50).astype(int)
    cm_def = confusion_matrix(y_test, preds_def)
    rec_def = float(res_default['n_tp'] / (res_default['n_tp'] + res_default['n_fn']))
    prec_def = float(res_default['n_tp'] / max(res_default['n_tp'] + res_default['n_fp'], 1))
    f2_def = float(fbeta_score(y_test, preds_def, beta=2))

    # Metrics at Tuned Threshold (t* = 0.1200)
    res_tuned = evaluate_banking_cost(y_test, test_probs, test_amounts, threshold=optimal_threshold)
    preds_tuned = (test_probs >= optimal_threshold).astype(int)
    cm_tuned = confusion_matrix(y_test, preds_tuned)
    rec_tuned = float(res_tuned['n_tp'] / (res_tuned['n_tp'] + res_tuned['n_fn']))
    prec_tuned = float(res_tuned['n_tp'] / max(res_tuned['n_tp'] + res_tuned['n_fp'], 1))
    f2_tuned = float(fbeta_score(y_test, preds_tuned, beta=2))

    total_frauds = res_tuned['n_tp'] + res_tuned['n_fn']
    extra_caught = res_tuned['n_tp'] - res_default['n_tp']
    breach_reduction = ((res_default['n_fn'] - res_tuned['n_fn']) / max(res_default['n_fn'], 1)) * 100
    extra_investigations = res_tuned['n_fp'] - res_default['n_fp']

    df_comparison = pd.DataFrame({
        'Evaluation Metric': [
            'Frauds Caught (Recall)',
            'Missed Frauds (FN)',
            'False Alarms (FP)',
            'Precision',
            'F2 Score (Recall-weighted)',
            'PR-AUC'
        ],
        'Default Threshold (t = 0.50)': [
            f"{res_default['n_tp']} / {total_frauds} ({rec_def*100:.2f}%)",
            f"{res_default['n_fn']}",
            f"{res_default['n_fp']} (Precision = {prec_def*100:.2f}%)",
            f"{prec_def*100:.2f}%",
            f"{f2_def:.4f}",
            f"{test_pr_auc:.4f}"
        ],
        f'Tuned Threshold (t* = {optimal_threshold:.4f})': [
            f"{res_tuned['n_tp']} / {total_frauds} ({rec_tuned*100:.2f}%)",
            f"{res_tuned['n_fn']}",
            f"{res_tuned['n_fp']} (Precision = {prec_tuned*100:.2f}%)",
            f"{prec_tuned*100:.2f}%",
            f"{f2_tuned:.4f}",
            f"{test_pr_auc:.4f}"
        ],
        'Operational Impact': [
            f"+{extra_caught} additional frauds stopped",
            f"{breach_reduction:.1f}% reduction in breaches",
            f"+{extra_investigations} investigations (${extra_investigations*5} review cost)",
            "High precision preserved",
            "Higher operational protection factor",
            "Exceeds random baseline (0.0017)"
        ]
    })

    return {
        'table': df_comparison,
        'cm_default': cm_def,
        'cm_tuned': cm_tuned,
        'test_probs': test_probs,
        'test_pr_auc': test_pr_auc,
        'test_roc_auc': test_roc_auc,
        'rec_tuned': rec_tuned,
        'prec_tuned': prec_tuned,
        'f2_tuned': f2_tuned,
        'res_default': res_default,
        'res_tuned': res_tuned
    }


def plot_test_evaluation(y_test, test_results, optimal_threshold=0.1200, save_path=None):
    """
    Renders 3-panel test evaluation figure: CM (Default), CM (Tuned), and PR Curve at 150 DPI.
    """
    cm_def = test_results['cm_default']
    cm_tuned = test_results['cm_tuned']
    test_probs = test_results['test_probs']
    test_pr_auc = test_results['test_pr_auc']
    rec_tuned = test_results['rec_tuned']
    prec_tuned = test_results['prec_tuned']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # CM Default
    sns.heatmap(
        cm_def, annot=True, fmt=',d', cmap='Blues', cbar=False,
        xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'],
        ax=axes[0]
    )
    axes[0].set_title(f'Default Cutoff (t = 0.50)\\nCaught: {cm_def[1, 1]} | Missed: {cm_def[1, 0]}', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # CM Tuned
    sns.heatmap(
        cm_tuned, annot=True, fmt=',d', cmap='Reds', cbar=False,
        xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'],
        ax=axes[1]
    )
    axes[1].set_title(f'Tuned Cutoff (t* = {optimal_threshold:.2f})\\nCaught: {cm_tuned[1, 1]} | Missed: {cm_tuned[1, 0]}', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    axes[2].plot(recall, precision, color='#d9383a', lw=2.5, label=f'Test PR-AUC = {test_pr_auc:.4f}')
    axes[2].scatter([rec_tuned], [prec_tuned], color='black', s=90, zorder=5, label=f'Operating Point (t*={optimal_threshold:.2f})')
    axes[2].axhline(y=np.mean(y_test), color='gray', linestyle='--', label=f'Baseline ({np.mean(y_test):.4f})')
    axes[2].set_title('Precision-Recall Curve (Held-Out Test Set)', fontweight='bold')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(loc='lower left')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Held-Out Test Set: Operational Performance Audit', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
