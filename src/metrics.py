"""
Metrics calculation and visualization utilities.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def print_metrics(metrics: Dict[str, float]):
    """Print metrics in formatted way."""
    print("\n" + "="*50)
    print("Metrics:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:15s}: {value:.4f}")
    print("="*50)
