# This module creates stratified train, validation, and test sets. 
# The preprocessing transformer is fitted only on the training partition.
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ann.config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


CATEGORICAL_FEATURES = ["Geography", "Gender"]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def build_preprocessor():
    """Build a reusable transformer for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_data(X, y):
    """Split data and fit preprocessing on the training data only."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed.astype("float32"),
        X_val_processed.astype("float32"),
        X_test_processed.astype("float32"),
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
    )