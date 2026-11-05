"""
train.py: 5-Fold Stratified Cross-Validation Benchmark and Champion Model Training.
Reproduces all 9 architectures and strategies from the Presentation Master Guide.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, auc, fbeta_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

from .features import build_preprocessor, create_pipeline


def calculate_scale_pos_weight(y_train):
    """
    Computes exact negative-to-positive ratio for XGBoost gradient stabilization:
    scale_pos_weight = N_neg / N_pos ≈ 576.
    """
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    return float(n_neg / max(n_pos, 1))


def get_benchmark_models(scale_pos_weight, random_state=42):
    """
    Constructs the 9 model pipelines specified in Section 4 of the Master Guide.
    """
    pre = build_preprocessor()

    models = {
        'Dummy (Majority Class)': create_pipeline(
            DummyClassifier(strategy='most_frequent'),
            preprocessor=pre
        ),
        'Logistic Regression (Balanced)': create_pipeline(
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
            preprocessor=pre
        ),
        'Random Forest (No Resampling)': create_pipeline(
            RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=random_state, n_jobs=1),
            preprocessor=pre
        ),
        'Random Forest + SMOTE': create_pipeline(
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=1),
            sampler=SMOTE(sampling_strategy=0.1, random_state=random_state),
            preprocessor=pre
        ),
        'Random Forest + RUS': create_pipeline(
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=1),
            sampler=RandomUnderSampler(sampling_strategy=0.1, random_state=random_state),
            preprocessor=pre
        ),
        'XGBoost (Unweighted / Default)': create_pipeline(
            xgb.XGBClassifier(
                n_estimators=60,
                learning_rate=0.3,
                max_depth=6,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1
            ),
            preprocessor=pre
        ),
        'XGBoost (scale_pos_weight)': create_pipeline(
            xgb.XGBClassifier(
                n_estimators=60,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1
            ),
            preprocessor=pre
        ),
        'XGBoost + SMOTE': create_pipeline(
            xgb.XGBClassifier(
                n_estimators=60,
                learning_rate=0.1,
                max_depth=5,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1
            ),
            sampler=SMOTE(sampling_strategy=0.1, random_state=random_state),
            preprocessor=pre
        ),
        'XGBoost (Tuned via RandomizedSearch)': create_pipeline(
            xgb.XGBClassifier(
                n_estimators=80,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1
            ),
            preprocessor=pre
        )
    }
    return models


def run_stratified_cv(models, X_train, y_train, n_splits=5, random_state=42):
    """
    Executes 5-Fold Stratified Cross-Validation on all benchmark models.
    Returns:
        df_results: pandas DataFrame formatted with Mean ± Std for PR-AUC, ROC-AUC, F2.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for name, pipeline in models.items():
        pr_aucs = []
        roc_aucs = []
        f2_scores = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_probs = pipeline.predict_proba(X_va)[:, 1]

            # PR-AUC
            pr_auc = average_precision_score(y_va, y_probs)
            # ROC-AUC
            fpr, tpr, _ = roc_curve(y_va, y_probs)
            roc_auc = auc(fpr, tpr)
            # F2 score at standard t=0.50 cutoff
            preds = (y_probs >= 0.50).astype(int)
            f2 = fbeta_score(y_va, preds, beta=2, zero_division=0)

            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
            f2_scores.append(f2)

        results.append({
            'Model Architecture & Strategy': name,
            'PR-AUC (mean +/- std)': f"{np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}",
            'ROC-AUC (mean +/- std)': f"{np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}",
            'F2 Score (mean +/- std)': f"{np.mean(f2_scores):.4f} +/- {np.std(f2_scores):.4f}",
            'raw_pr_auc': np.mean(pr_aucs),
            'raw_f2': np.mean(f2_scores)
        })

    return pd.DataFrame(results)


def train_champion_model(X_train, y_train, scale_pos_weight, random_state=42):
    """
    Trains the champion Tuned XGBoost model on the complete 70% Train partition.
    """
    champion_pipeline = create_pipeline(
        xgb.XGBClassifier(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=1
        )
    )
    champion_pipeline.fit(X_train, y_train)
    return champion_pipeline
