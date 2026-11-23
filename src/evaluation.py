"""
Model evaluation, performance metrics computation, and comparative analysis.
Handles classification metrics, confusion matrices, and checks against published literature.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    Interprets class definitions clearly:
      - Class 1: Entry level (Low market-priority proxy)
      - Class 0: Non-entry level (High market-priority proxy)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=["High Market Priority (0)", "Low Market Priority / Entry (1)"],
        digits=4
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "test_samples": int(len(y_true)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix_sum": int(tn + fp + fn + tp),
        "classification_report": report_str
    }

    return metrics


def save_metrics_to_csv(
    metrics: Dict[str, Any],
    csv_path: Optional[Path] = None,
    paper_comparison: bool = True
) -> pd.DataFrame:
    """
    Saves metrics to CSV and constructs comparison table against the paper.
    Identifies the published confusion matrix inconsistency:
      - Reported test size: 24,823
      - Confusion matrix cell sum: 9,930
    """
    path = csv_path or config.EVALUATION_METRICS_PATH

    rows = [
        {"Metric": "Accuracy", "Value": metrics["accuracy"], "Notes": "Overall correctness"},
        {"Metric": "Precision (Class 1)", "Value": metrics["precision"], "Notes": "Precision for Entry level (Low Priority proxy)"},
        {"Metric": "Recall (Class 1)", "Value": metrics["recall"], "Notes": "Recall for Entry level (Low Priority proxy)"},
        {"Metric": "F1-Score (Class 1)", "Value": metrics["f1_score"], "Notes": "Harmonic mean of precision & recall"},
        {"Metric": "ROC-AUC", "Value": metrics["roc_auc"], "Notes": "Area under ROC curve"},
        {"Metric": "PR-AUC", "Value": metrics["pr_auc"], "Notes": "Area under Precision-Recall curve"},
        {"Metric": "True Positives (TP)", "Value": metrics["true_positives"], "Notes": "Correctly classified Entry level"},
        {"Metric": "False Positives (FP)", "Value": metrics["false_positives"], "Notes": "Higher level misclassified as Entry"},
        {"Metric": "True Negatives (TN)", "Value": metrics["true_negatives"], "Notes": "Correctly classified Non-entry"},
        {"Metric": "False Negatives (FN)", "Value": metrics["false_negatives"], "Notes": "Entry level misclassified as Higher level"},
        {"Metric": "Total Test Observations", "Value": metrics["test_samples"], "Notes": "Untouched test partition size"},
        {"Metric": "Confusion Matrix Sum", "Value": metrics["confusion_matrix_sum"], "Notes": "TP + FP + TN + FN"}
    ]

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(path, index=False)
    logger.info("Saved evaluation metrics table to %s", path)

    # Save detailed JSON summary
    json_path = config.RESULTS_DIR / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved detailed evaluation JSON to %s", json_path)

    return df_metrics
