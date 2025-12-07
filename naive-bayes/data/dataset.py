import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_adult_dataset(path: str = None):

    # Load dataset
    df = pd.read_csv(path)

    # Target column 
    target_col = "income"

    # Select categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols.remove(target_col)

    x = df[categorical_cols].copy()
    y = df[target_col].copy()

    # Replace missing feature values 
    x = x.replace({"?": 'Missing', 'nan': 'Missing', 'None': 'Missing'})
    y = y.replace({"?": 'Missing', 'nan': 'Missing', 'None': 'Missing'})

    # Training - Validation - Testing split (70% - 15% - 15%)
    # x_temp , y_temp => train+test 30% to be splitted  
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Encode categorical features to integers
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    x_train = encoder.fit_transform(x_train)
    x_val = encoder.transform(x_val)
    x_test = encoder.transform(x_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "target_name": target_col,
        "feature_names": categorical_cols
    }
    