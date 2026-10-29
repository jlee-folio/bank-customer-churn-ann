# This module is the entry point that connects all modules.
# It sets random seeds, trains the network, evaluates the untouched test set,
# and saves the fitted artifacts.
import json
import random
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from churn_ann.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    RANDOM_STATE,
)
from churn_ann.data import load_data
from src.evaluate import evaluate_model, save_metrics
from churn_ann.model import build_ann
from src.preprocessing import prepare_data
from src.generate_report import generate_report

def main():
    # Reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # Optional: request deterministic TensorFlow operations.
    # Some operations/devices may not support deterministic execution.
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and prepare partitions
    X, y = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
    ) = prepare_data(X, y)

    # Build model
    model = build_ann(input_dim=X_train.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    # Train using validation data for model selection
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Evaluate the final selected weights on the held-out test set
    test_probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    metrics = evaluate_model(y_test, test_probabilities)
    
    with (ARTIFACTS_DIR / "training_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )
    # Generate statistical report, JSON metrics, and charts.
    reports_dir = ARTIFACTS_DIR.parent / "reports"

    generate_report(
        y_true=y_test,
        y_prob=test_probabilities,
        metrics=metrics,
        history=history.history,
        reports_dir=reports_dir,
    )
    print("\nTest-set evaluation")

    # Save model and preprocessing object together as separate artifacts
    model.save(ARTIFACTS_DIR / "ann_model.keras")
    joblib.dump(
        preprocessor,
        ARTIFACTS_DIR / "preprocessor.joblib",
    )

    save_metrics(
        metrics,
        ARTIFACTS_DIR / "metrics.json",
    )

    with (ARTIFACTS_DIR / "training_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )

    print("\nTest-set evaluation")
    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ):
        print(f"{name}: {metrics[name]:.4f}")

    print("\nConfusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()