#This module defines the ANN architecture separately from the training workflow.
import tensorflow as tf

from churn_ann.config import LEARNING_RATE


def build_ann(input_dim, learning_rate=LEARNING_RATE):
    """Build and compile a binary classification neural network."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )

    return model