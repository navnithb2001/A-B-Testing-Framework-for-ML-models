"""
Metrics calculation for ML models
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score
)


class MetricsCalculator:
    """Calculate various ML performance metrics"""
    
    @staticmethod
    def classification_metrics(y_true: List[int], y_pred: List[int], 
                               y_prob: List[float] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC-ROC)
        
        Returns:
            Dictionary with accuracy, precision, recall, F1, confusion matrix, etc.
        """
        if len(y_true) == 0:
            return {
                "error": "No data to calculate metrics",
                "sample_size": 0
            }
        
        metrics = {
            "sample_size": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            })
        
        # AUC-ROC if probabilities provided
        if y_prob is not None and len(y_prob) > 0:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            except:
                metrics["auc_roc"] = None
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            Dictionary with MSE, RMSE, MAE, R2, etc.
        """
        if len(y_true) == 0:
            return {
                "error": "No data to calculate metrics",
                "sample_size": 0
            }
        
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            "sample_size": len(y_true),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred))
        }
        
        # Mean Absolute Percentage Error
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        non_zero = y_true_arr != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100
            metrics["mape"] = float(mape)
        
        return metrics
    
    @staticmethod
    def latency_metrics(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics
        
        Args:
            latencies: List of latency measurements in milliseconds
        
        Returns:
            Dictionary with mean, median, p50, p95, p99 latencies
        """
        if len(latencies) == 0:
            return {
                "error": "No latency data",
                "sample_size": 0
            }
        
        latencies_arr = np.array(latencies)
        
        return {
            "sample_size": len(latencies),
            "mean_ms": float(np.mean(latencies_arr)),
            "median_ms": float(np.median(latencies_arr)),
            "std_ms": float(np.std(latencies_arr)),
            "min_ms": float(np.min(latencies_arr)),
            "max_ms": float(np.max(latencies_arr)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99))
        }
    
    @staticmethod
    def conversion_rate_metrics(conversions: List[int]) -> Dict[str, Any]:
        """
        Calculate conversion rate metrics
        
        Args:
            conversions: List of 1s (conversion) and 0s (no conversion)
        
        Returns:
            Dictionary with conversion rate, total conversions, etc.
        """
        if len(conversions) == 0:
            return {
                "error": "No conversion data",
                "sample_size": 0
            }
        
        conversions_arr = np.array(conversions)
        total_conversions = int(np.sum(conversions_arr))
        conversion_rate = float(np.mean(conversions_arr))
        
        return {
            "sample_size": len(conversions),
            "total_conversions": total_conversions,
            "conversion_rate": conversion_rate,
            "conversion_rate_percent": conversion_rate * 100
        }
