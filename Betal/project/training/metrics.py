"""
Metrics calculation.
Computes Accuracy, Precision, Recall, F1 Score, and ROC AUC.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy.typing as npt

def calculate_metrics(y_true: npt.NDArray, y_pred: npt.NDArray, y_prob: npt.NDArray) -> dict:
    """
    Calculate classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0 # Handle case where only 1 class is present in batch
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
