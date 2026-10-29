# # This test uses synthetic data to verify that the pipeline runs successfully.
import numpy as np
import pandas as pd

from churn_ann.preprocessing import prepare_data


def test_preprocessing_produces_valid_partitions():
    n = 200

    X = pd.DataFrame(
        {
            "CreditScore": np.linspace(400, 800, n),
            "Geography": ["France", "Spain", "Germany", "France"] * 50,
            "Gender": ["Male", "Female"] * 100,
            "Age": np.tile([25, 35, 45, 55], 50),
            "Tenure": np.tile([1, 3, 5, 7], 50),
            "Balance": np.linspace(0, 200000, n),
            "NumOfProducts": np.tile([1, 2], 100),
            "HasCrCard": np.tile([0, 1], 100),
            "IsActiveMember": np.tile([0, 1], 100),
            "EstimatedSalary": np.linspace(20000, 150000, n),
        }
    )

    y = pd.Series(np.tile([0, 1], 100))

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
    ) = prepare_data(X, y)

    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    assert len(X_train) + len(X_val) + len(X_test) == n
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()

    assert preprocessor is not None