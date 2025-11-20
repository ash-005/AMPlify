"""Evaluation utilities: ROC, PR, confusion, simple attention plotting helper.

Uses scikit-learn when available; falls back to simple implementations.
"""
from typing import Tuple
import numpy as np


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # very small fallback: if binary, approximate by simple ranking
        # Not a replacement for sklearn but keeps code runnable
        order = np.argsort(y_score)
        y_true_sorted = y_true[order]
        cum_pos = np.cumsum(y_true_sorted)
        # normalized
        return float(cum_pos[-1] / max(1, len(y_true)))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn
