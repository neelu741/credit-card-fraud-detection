"""
explain.py: SHAP TreeExplainer Model Interpretability.
Generates global beeswarm summary and local waterfall plots.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import shap


def explain_model(pipeline, X_sample, y_sample=None, probs_sample=None, save_dir="reports/figures"):
    """
    Computes SHAP values using TreeExplainer on the transformed feature space.
    Saves shap_summary.png at 150 DPI and returns explainer & shap_values.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract underlying fitted XGBoost model and preprocessor
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X_trans = preprocessor.transform(X_sample)
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_trans_df)

    # 1. Global Summary Beeswarm Plot
    plt.figure(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values, X_trans_df, show=False, max_display=10)
    plt.title('Global Feature Drivers: V14, V4, V12, V10, V17 Dominate Predictions', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'shap_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Local Waterfall Plots (if samples and labels provided)
    if y_sample is not None and probs_sample is not None:
        tp_indices = [i for i, (y_val, p_val) in enumerate(zip(y_sample, probs_sample)) if y_val == 1 and p_val >= 0.12]
        if len(tp_indices) > 0:
            tp_idx = tp_indices[0]
            plt.figure(figsize=(9, 4.5), dpi=150)
            shap.plots.waterfall(shap_values[tp_idx], max_display=8, show=False)
            plt.title(f'Local Waterfall Audit: True Positive (Risk Score = {probs_sample[tp_idx]:.4f} > 0.12)', fontweight='bold')
            plt.tight_layout()
            waterfall_path = os.path.join(save_dir, 'shap_waterfall_tp.png')
            plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
            plt.close()

    return explainer, shap_values
