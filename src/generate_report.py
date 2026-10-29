
"""Generate statistical documentation and charts for ANN evaluation."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Supports saving charts without a GUI.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def _save_confusion_matrix(metrics, output_path):
    """Plot the confusion matrix using actual class counts."""
    cm = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of customers")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Stay (0)", "Churn (1)"],
        yticklabels=["Stay (0)", "Churn (1)"],
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Customer Churn Confusion Matrix",
    )

    midpoint = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > midpoint else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curve(y_true, y_prob, metrics, output_path):
    """Plot ROC performance against the random-classifier baseline."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        fpr,
        tpr,
        label=f"ANN (ROC-AUC = {metrics['roc_auc']:.4f})",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random baseline")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Recall)",
        title="Receiver Operating Characteristic (ROC)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_precision_recall_curve(y_true, y_prob, metrics, output_path):
    """Plot precision versus recall across probability thresholds."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(np.asarray(y_true) == 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"ANN (AP = {metrics['pr_auc']:.4f})",
    )
    ax.axhline(
        prevalence,
        linestyle="--",
        color="grey",
        label=f"Positive-class prevalence = {prevalence:.3f}",
    )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_training_history(history, output_path):
    """Plot training and validation loss and selected metrics."""
    history = dict(history)
    if not history:
        return False

    preferred = [
        ("loss", "val_loss", "Loss"),
        ("roc_auc", "val_roc_auc", "ROC-AUC"),
        ("pr_auc", "val_pr_auc", "PR-AUC / Average Precision"),
        ("accuracy", "val_accuracy", "Accuracy"),
    ]

    available = [
        item
        for item in preferred
        if item[0] in history or item[1] in history
    ]
    if not available:
        return False

    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(7, 2.5 * len(available)),
        squeeze=False,
    )

    for ax, (train_key, val_key, title) in zip(
        axes[:, 0], available
    ):
        if train_key in history:
            ax.plot(
                range(1, len(history[train_key]) + 1),
                history[train_key],
                label=f"Training {title}",
            )
        if val_key in history:
            ax.plot(
                range(1, len(history[val_key]) + 1),
                history[val_key],
                label=f"Validation {title}",
            )

        ax.set(
            xlabel="Epoch",
            ylabel=title,
            title=f"{title} During Training",
        )
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_report(
    y_true,
    y_prob,
    metrics,
    history,
    reports_dir,
):
    """Save charts, JSON metrics, and a Markdown evaluation report."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()

    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must have equal, nonzero lengths.")
    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("This report generator expects binary labels 0 and 1.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC and PR curves require both classes in the test set.")

    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    if cm.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")

    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    prevalence = float(np.mean(y_true == 1))
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0

    chart_names = {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "precision_recall_curve": "precision_recall_curve.png",
        "training_history": "training_history.png",
    }

    _save_confusion_matrix(
        metrics, reports_dir / chart_names["confusion_matrix"]
    )
    _save_roc_curve(
        y_true, y_prob, metrics, reports_dir / chart_names["roc_curve"]
    )
    _save_precision_recall_curve(
        y_true,
        y_prob,
        metrics,
        reports_dir / chart_names["precision_recall_curve"],
    )
    has_history_plot = _save_training_history(
        history, reports_dir / chart_names["training_history"]
    )

    # Keep both the requested PR-AUC metric and its sklearn definition explicit.
    report_metrics = {
        **metrics,
        "test_samples": int(n),
        "actual_stay": int(tn + fp),
        "actual_churn": int(fn + tp),
        "predicted_stay": int(tn + fn),
        "predicted_churn": int(fp + tp),
        "churn_prevalence": prevalence,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "pr_auc_definition": "scikit-learn average_precision_score",
        "generated_at": datetime.now().astimezone().isoformat(),
    }

    with (reports_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report_metrics, file, indent=2, allow_nan=False)

    threshold = metrics["threshold"]
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    roc_auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]

    if recall < 0.60:
        recall_assessment = (
            "Recall is relatively low at the selected threshold. "
            "Review threshold tuning on validation data if identifying "
            "more churners is a business priority."
        )
    else:
        recall_assessment = (
            "Recall should be assessed against the business's required "
            "coverage of churners and the cost of retention interventions."
        )

    if roc_auc >= 0.85:
        auc_assessment = "The ROC-AUC indicates strong ranking discrimination on this test set."
    elif roc_auc >= 0.70:
        auc_assessment = "The ROC-AUC indicates moderate ranking discrimination on this test set."
    else:
        auc_assessment = "The ROC-AUC suggests that ranking performance warrants further investigation."

    recommendations = [
        "Tune the classification threshold using validation data; do not select it using the final test set.",
        "Compare against logistic regression and tree-based baselines using identical data splits.",
        "Estimate intervention costs and retention benefits before selecting an operating threshold.",
        "Assess probability calibration and evaluate performance on later customer cohorts.",
    ]

    lines = [
        "# Customer Churn ANN — Statistical Evaluation Report",
        "",
        f"**Generated:** {report_metrics['generated_at']}",
        "",
        "## 1. Executive summary",
        "",
        f"The model was evaluated on **{n:,} held-out test observations**. "
        f"Accuracy was **{accuracy:.2%}**, ROC-AUC was **{roc_auc:.4f}**, "
        f"precision was **{precision:.2%}**, and recall was **{recall:.2%}** "
        f"at a classification threshold of **{threshold:.2f}**.",
        "",
        f"{auc_assessment} {recall_assessment}",
        "",
        "These results describe performance on this test set; they do not "
        "by themselves establish statistical significance or business impact.",
        "",
        "## 2. Evaluation metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {accuracy:.4f} ({accuracy:.2%}) |",
        f"| Precision | {precision:.4f} ({precision:.2%}) |",
        f"| Recall / Sensitivity | {recall:.4f} ({recall:.2%}) |",
        f"| F1 score | {f1:.4f} |",
        f"| ROC-AUC | {roc_auc:.4f} |",
        f"| Average precision (PR-AUC field) | {pr_auc:.4f} |",
        f"| Specificity | {specificity:.4f} ({specificity:.2%}) |",
        f"| Classification threshold | {threshold:.2f} |",
        "",
        "Average precision is calculated with scikit-learn's "
        "`average_precision_score`; it is not the trapezoidal integral "
        "of the precision-recall curve.",
        "",
        "## 3. Confusion matrix",
        "",
        "| Actual class | Predicted stay (0) | Predicted churn (1) | Total |",
        "|---|---:|---:|---:|",
        f"| Stay (0) | {tn:,} | {fp:,} | {tn + fp:,} |",
        f"| Churn (1) | {fn:,} | {tp:,} | {fn + tp:,} |",
        f"| **Total** | **{tn + fn:,}** | **{fp + tp:,}** | **{n:,}** |",
        "",
        f"- False-positive rate: **{fpr:.2%}**.",
        f"- False-negative rate: **{fnr:.2%}**.",
        f"- Churn prevalence in test set: **{prevalence:.2%}**.",
        "",
        f"At this threshold, the model identified {tp:,} of {tp + fn:,} "
        f"actual churners and missed {fn:,}. It flagged {fp + tp:,} "
        "customers as potential churners.",
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "## 4. Diagnostic charts",
        "",
        "### ROC curve",
        "",
        "![ROC curve](roc_curve.png)",
        "",
        "### Precision-recall curve",
        "",
        "![Precision-recall curve](precision_recall_curve.png)",
        "",
    ]

    if has_history_plot:
        lines.extend([
            "### Training and validation curves",
            "",
            "![Training history](training_history.png)",
            "",
            "Review the curves for divergence between training and validation "
            "performance. Early stopping alone does not prove that overfitting "
            "has been eliminated.",
            "",
        ])

    lines.extend([
        "## 5. Interpretation and limitations",
        "",
        "- Precision and recall reflect the selected classification threshold.",
        "- ROC-AUC measures ranking discrimination, not probability calibration.",
        "- Average precision should be interpreted relative to churn prevalence "
        "and suitable baseline models.",
        "- The results do not establish causality or the effectiveness of "
        "retention interventions.",
        "- Confirm that preprocessing was fitted only on training data and "
        "that the test set was not used for model or threshold selection.",
        "",
        "## 6. Recommendations",
        "",
    ])
    lines.extend(f"{i}. {item}" for i, item in enumerate(recommendations, 1))
    lines.extend([
        "",
        "## 7. Reproducibility",
        "",
        f"- Test observations: {n:,}",
        f"- Positive-class prevalence: {prevalence:.4f}",
        f"- Decision threshold: {threshold:.2f}",
        "- Detailed metrics: `metrics.json`",
        "- Model and preprocessing artifacts should be versioned separately.",
        "",
        "---",
        "",
        "This report was generated automatically from the current run's test "
        "labels, predicted probabilities, evaluation metrics, and training history.",
        "",
    ])

    (reports_dir / "model_evaluation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Statistical report saved to: {reports_dir.resolve()}")
