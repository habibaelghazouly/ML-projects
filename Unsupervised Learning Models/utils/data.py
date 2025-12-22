import os
import pandas as pd
import numpy as np


def load_breast_cancer_kagglehub():
    """
    Load the Breast Cancer Wisconsin dataset from KaggleHub with robust handling.

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels (1 for Malignant, 0 for Benign) or None
        feature_names (list): List of column names used as features

    Raises:
        FileNotFoundError: If no CSV files found in dataset path
        ValueError: If dataset is empty after cleaning or has invalid format
    """
    import kagglehub

    # Download dataset
    path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

    # Find all CSV files
    csv_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in kagglehub path: {path}")

    # Try to find CSV containing 'diagnosis' column
    df = None
    chosen_file = None

    for fp in csv_files:
        try:
            tmp = pd.read_csv(fp)
            if "diagnosis" in [col.lower() for col in tmp.columns]:
                df = tmp
                chosen_file = fp
                break
        except Exception as e:
            print(f"Warning: Could not read {fp}: {e}")
            continue

    # If no file with 'diagnosis' found, use first readable CSV
    if df is None:
        for fp in csv_files:
            try:
                df = pd.read_csv(fp)
                chosen_file = fp
                break
            except Exception as e:
                print(f"Warning: Could not read {fp}: {e}")
                continue

    if df is None:
        raise FileNotFoundError("Could not read any CSV files from dataset")

    print(f"Loaded CSV: {chosen_file}")
    print(f"Initial shape: {df.shape}")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Drop columns that are entirely NaN
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        print(f"Dropping {len(null_cols)} columns with all NaN values: {null_cols}")
        df = df.drop(columns=null_cols)

    # Handle target column extraction
    y = None
    target_col = None

    # Case-insensitive search for diagnosis column
    for col in df.columns:
        if col.lower() == "diagnosis":
            target_col = col
            break

    if target_col:
        # Extract and encode target variable
        diagnosis_values = df[target_col].dropna().unique()
        print(f"Found diagnosis column with values: {diagnosis_values}")

        # Handle different encoding schemes
        if df[target_col].dtype == "object":
            # Map string labels to binary
            y = df[target_col].map({"M": 1, "B": 0})

            # Check for unmapped values
            if y.isnull().any():
                unmapped = df[target_col][y.isnull()].unique()
                print(f"Warning: Unmapped diagnosis values found: {unmapped}")
                # Try alternative mappings
                y = df[target_col].map(
                    {"M": 1, "B": 0, "Malignant": 1, "Benign": 0, "1": 1, "0": 0}
                )
        else:
            # Numeric labels
            y = df[target_col]

        y = y.to_numpy()
        df = df.drop(columns=[target_col])

    # Remove non-informative columns (ID, unnamed columns)
    cols_to_drop = []
    for col in df.columns:
        if col.lower() == "id" or col.lower().startswith("unnamed"):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Dropping non-informative columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns: {non_numeric_cols}")
        print("Attempting to convert or drop these columns...")

        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"  Converted {col} to numeric")
            except:
                print(f"  Dropping {col} (cannot convert to numeric)")
                df = df.drop(columns=[col])

    # Handle missing values in features
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print(f"\nMissing values found in {(missing_counts > 0).sum()} columns:")
        for col, count in missing_counts[missing_counts > 0].items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.2f}%)")

        # Strategy: Drop rows with any missing values (conservative approach)
        initial_rows = len(df)
        df = df.dropna(axis=0, how="any")
        rows_dropped = initial_rows - len(df)

        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing values")

            # Align target variable if it exists
            if y is not None:
                y = y[df.index]

    # Validate we still have data
    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    if len(df.columns) == 0:
        raise ValueError("No feature columns remaining after cleaning")

    # Convert to numpy arrays
    feature_names = list(df.columns)
    X = df.astype(float).to_numpy()

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    if y is not None:
        print(f"Target distribution: Malignant={np.sum(y==1)}, Benign={np.sum(y==0)}")

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
