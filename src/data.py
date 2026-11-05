"""
data.py: Ingestion, auditing, duplicate pruning, and stratified splitting.
Strictly preserves leakage defense: 70% Train, 15% Validation, 15% Test.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path="data/creditcard.csv"):
    """
    Load raw transaction dataset. Falls back to search in parent directories.
    """
    candidate_paths = [
        file_path,
        os.path.join("..", file_path),
        os.path.join(os.path.dirname(__file__), "..", file_path),
        "creditcard.csv"
    ]
    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            resolved_path = p
            break
            
    if resolved_path is None:
        raise FileNotFoundError(f"Dataset not found at '{file_path}'. Please place creditcard.csv in data/")
        
    df = pd.read_csv(resolved_path)
    return df


def audit_data(df):
    """
    Performs data integrity check: dimensions, missing values, duplicate counts.
    """
    n_rows, n_cols = df.shape
    n_nulls = int(df.isnull().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    fraud_count = int(df['Class'].sum())
    fraud_pct = float(df['Class'].mean() * 100)
    
    audit_summary = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'null_values': n_nulls,
        'duplicate_rows': n_duplicates,
        'fraud_count': fraud_count,
        'fraud_percentage': fraud_pct
    }
    return audit_summary


def prune_duplicates(df):
    """
    Drops duplicate rows prior to partitioning to guarantee zero train-test leakage.
    (Prunes exactly 1,081 duplicates from 284,807 -> 283,726).
    """
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df_clean


def split_data(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Executes a 70 / 15 / 15 stratified split.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, val_amounts, test_amounts
    """
    X = df.drop(columns=['Class'])
    y = df['Class'].values
    amounts = df['Amount'].values

    # Step 1: 70% Train, 30% Temp (Val + Test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp, _, amounts_temp = train_test_split(
        X, y, amounts,
        test_size=temp_size,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Split Temp into equal Validation (15%) and Held-Out Test (15%)
    val_relative_ratio = val_size / temp_size
    X_val, X_test, y_val, y_test, val_amounts, test_amounts = train_test_split(
        X_temp, y_temp, amounts_temp,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        val_amounts, test_amounts
    )
