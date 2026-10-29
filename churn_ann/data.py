# Load the CSV and check that the expected columns and target values exist.
import pandas as pd

from churn_ann.config import DATASET_PATH, FEATURES, TARGET


def load_data(path=DATASET_PATH):
    """Load the customer churn dataset with basic schema validation."""
    path = path.expanduser() if hasattr(path, "expanduser") else path

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place Churn_Modelling.csv in the data/ directory."
        )

    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values detected. Add an explicit imputation "
            "strategy before training."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1.")

    X = df[FEATURES].copy()
    y = df[TARGET].astype("int32").copy()

    return X, y