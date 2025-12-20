import os
import numpy as np
import pandas as pd

def load_breast_cancer_kagglehub():
    import kagglehub

    path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
    csv_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in kagglehub path: {path}")

    # Load candidate CSVs and choose one that contains diagnosis column
    df = None
    chosen = None
    for fp in csv_files:
        tmp = pd.read_csv(fp)
        cols = [c.lower() for c in tmp.columns]
        if "diagnosis" in cols:
            df = tmp
            chosen = fp
            break

    if df is None:
        chosen = csv_files[0]
        df = pd.read_csv(chosen)

    print("Loaded CSV:", chosen)
    df = df.copy()
    
    # 1) Drop columns that are entirely NaN (e.g., "Unnamed: 32")
    df = df.dropna(axis=1, how="all")

    # 2) Drop any remaining rows that still contain NaNs (usually none here)
    df = df.dropna(axis=0, how="any")


    y = None
    if "diagnosis" in df.columns:
        # M -> 1, B -> 0
        y = df["diagnosis"].map({"M": 1, "B": 0}).to_numpy()
        df = df.drop(columns=["diagnosis"])

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])

    feature_names = list(df.columns)
    X = df.to_numpy(dtype=float)

    return X, y, feature_names


def standardize_fit_transform(X: np.ndarray, eps: float = 1e-12):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < eps, 1.0, std)
    Xs = (X - mean) / std
    scaler = {"mean": mean, "std": std}
    return Xs, scaler


def standardize_transform(X: np.ndarray, scaler: dict):
    mean, std = scaler["mean"], scaler["std"]
    return (X - mean) / std


def standardize_inverse_transform(Xs: np.ndarray, scaler: dict):
    mean, std = scaler["mean"], scaler["std"]
    return Xs * std + mean
