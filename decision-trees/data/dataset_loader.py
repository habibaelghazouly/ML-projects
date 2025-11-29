from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


def load_bcw_dataset(test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads the Breast Cancer Wisconsin dataset and splits it into
    train (70%), validation (15%), and test (15%) sets using stratified sampling.

    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
        feature_names
        target_names
    """

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # First split: temp test (15%) and temp train_val (85%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Compute validation size relative to remaining data (train+val)
    relative_val_size = val_size / (1 - test_size)

    # Second split: train (70%) and validation (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, target_names)
