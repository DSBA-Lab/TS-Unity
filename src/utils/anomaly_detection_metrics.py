"""
Anomaly Detection Metrics Module

This module provides comprehensive metrics for evaluating anomaly detection models,
including point-wise metrics, sequence-based metrics, and threshold optimization methods.

Classes:
    AnomalyMetrics: Basic evaluation metrics (MAE, MSE, RMSE, etc.)
    PointMetrics: Point-wise evaluation metrics (F1, precision, recall, etc.)
    ThresholdOptimization: Methods for finding optimal thresholds
    SequenceMetrics: Sequence-based evaluation metrics
    AdvancedMetrics: Advanced evaluation methods and techniques

The module maintains backward compatibility by providing function aliases for all
class methods, allowing existing code to continue working without modification.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, roc_curve, roc_auc_score, precision_score, 
    recall_score, average_precision_score, auc, precision_recall_curve
)
from collections import Counter
from typing import Tuple, List, Dict, Optional, Union
import logging

# Constants
EPSILON = 1e-5
DEFAULT_ANOMALY_THRESHOLD = 0.1
DEFAULT_ANOMALY_LABEL_THRESHOLD = 0.5
DEFAULT_Q_VALUE = 1e-3
DEFAULT_LEVEL = 0.02
DEFAULT_DISPLAY_FREQ = 1

logger = logging.getLogger(__name__)


class AnomalyMetrics:
    """Class containing all anomaly detection metrics and evaluation methods."""
    
    @staticmethod
    def rse(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Relative Squared Error."""
        numerator = np.sqrt(np.sum((true - pred) ** 2))
        denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
        return numerator / (denominator + EPSILON)
    
    @staticmethod
    def corr(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Correlation coefficient."""
        u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
        return (u / (d + EPSILON)).mean(-1)
    
    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(pred - true))
    
    @staticmethod
    def mse(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((pred - true) ** 2)
    
    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(AnomalyMetrics.mse(pred, true))
    
    @staticmethod
    def mape(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((pred - true) / (true + EPSILON)))
    
    @staticmethod
    def mspe(pred: np.ndarray, true: np.ndarray) -> float:
        """Calculate Mean Squared Percentage Error."""
        return np.mean(np.square((pred - true) / (true + EPSILON)))
    
    @staticmethod
    def calculate_all_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[float, ...]:
        """Calculate all basic metrics at once."""
        mae = AnomalyMetrics.mae(pred, true)
        mse = AnomalyMetrics.mse(pred, true)
        rmse = AnomalyMetrics.rmse(pred, true)
        mape = AnomalyMetrics.mape(pred, true)
        mspe = AnomalyMetrics.mspe(pred, true)
        return mae, mse, rmse, mape, mspe


class PointMetrics:
    """Point-wise evaluation metrics for anomaly detection."""
    
    @staticmethod
    def calc_point2point(predict: np.ndarray, actual: np.ndarray) -> Tuple[float, ...]:
        """
        Calculate point-wise metrics: F1, precision, recall, TP, TN, FP, FN.
        
        Args:
            predict: Predicted labels (0 or 1)
            actual: Actual labels (0 or 1)
            
        Returns:
            Tuple of (f1, precision, recall, TP, TN, FP, FN)
        """
        if predict.shape != actual.shape:
            raise ValueError("predict and actual must have the same shape")
        
        TP = np.sum(predict * actual)
        TN = np.sum((1 - predict) * (1 - actual))
        FP = np.sum(predict * (1 - actual))
        FN = np.sum((1 - predict) * actual)
        
        precision = TP / (TP + FP + EPSILON)
        recall = TP / (TP + FN + EPSILON)
        f1 = 2 * precision * recall / (precision + recall + EPSILON)
        
        return f1, precision, recall, TP, TN, FP, FN


class ThresholdOptimization:
    """Methods for finding optimal thresholds for anomaly detection."""
    
    @staticmethod
    def adjust_predicts(
        score: np.ndarray, 
        label: np.ndarray,
        threshold: Optional[float] = None,
        pred: Optional[np.ndarray] = None,
        calc_latency: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Calculate adjusted prediction labels using score, threshold, and label.
        
        Args:
            score: Anomaly scores
            label: Ground truth labels
            threshold: Threshold for anomaly detection
            pred: Pre-computed predictions (if available)
            calc_latency: Whether to calculate detection latency
            
        Returns:
            Adjusted predictions, optionally with latency
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0.0
        
        if pred is None:
            predict = score > threshold
        else:
            predict = pred.copy()
        
        actual = label > DEFAULT_ANOMALY_THRESHOLD
        anomaly_state = False
        anomaly_count = 0
        
        for i in range(len(score)):
            if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                # Backward adjustment for continuity
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
            elif not actual[i]:
                anomaly_state = False
            
            if anomaly_state:
                predict[i] = True
        
        if calc_latency:
            return predict, latency / (anomaly_count + EPSILON)
        return predict
    
    @staticmethod
    def pa_percentile(
        score: np.ndarray, 
        label: np.ndarray,
        threshold: Optional[float] = None,
        pred: Optional[np.ndarray] = None,
        K: int = 100,
        calc_latency: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Calculate adjusted predictions using percentile-based approach.
        
        Args:
            score: Anomaly scores
            label: Ground truth labels
            threshold: Threshold for anomaly detection
            pred: Pre-computed predictions
            K: Percentile threshold for segment detection
            calc_latency: Whether to calculate detection latency
            
        Returns:
            Adjusted predictions, optionally with latency
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same shape")
        
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0.0
        
        if pred is None:
            predict = score > threshold
        else:
            predict = pred.copy()
        
        actual = label > DEFAULT_ANOMALY_THRESHOLD
        anomaly_state = False
        anomaly_count = 0
        anomalies = []
        
        # Find anomaly segments
        for i in range(len(actual)):
            if actual[i]:
                if not anomaly_state:
                    anomaly_state = True
                    anomaly_count += 1
                    anomalies.append([i, i])
                else:
                    anomalies[-1][-1] = i
            else:
                anomaly_state = False
        
        # Apply percentile-based adjustment
        for start, end in anomalies:
            collect = Counter(predict[start:end + 1])[1]
            collect_ratio = collect / (end - start + 1)
            
            if collect_ratio * 100 >= K and collect > 0:
                predict[start:end + 1] = True
                latency += (end - start + 1) - collect
        
        if calc_latency:
            return predict, latency / (anomaly_count + EPSILON)
        return predict
    
    @staticmethod
    def bf_search(
        score: np.ndarray, 
        label: np.ndarray, 
        start: float, 
        end: Optional[float] = None, 
        step_num: int = 1, 
        display_freq: int = DEFAULT_DISPLAY_FREQ, 
        K: int = 0, 
        verbose: bool = True
    ) -> Tuple[Tuple[float, ...], float]:
        """
        Find best F1 score by searching optimal threshold.
        
        Args:
            score: Anomaly scores
            label: Ground truth labels
            start: Start threshold
            end: End threshold
            step_num: Number of search steps
            display_freq: Frequency of progress display
            K: Percentile parameter
            verbose: Whether to show progress
            
        Returns:
            Tuple of (best_metrics, best_threshold)
        """
        if end is None:
            end = start
            step_num = 1
        
        search_step = step_num
        search_range = end - start
        search_lower_bound = start
        
        if verbose:
            logger.info(f"Search range: {search_lower_bound} to {search_lower_bound + search_range}")
        
        threshold = search_lower_bound
        best_metrics = (-1.0, -1.0, -1.0)
        best_threshold = 0.0

        for i in range(search_step):
            threshold += search_range / float(search_step)
            target = SequenceMetrics.calc_seq(score, label, threshold, K=K, calc_latency=True)
            
            if target[0] > best_metrics[0]:
                best_threshold = threshold
                best_metrics = target
            
            if verbose and i % display_freq == 0:
                logger.info(f"Current threshold: {threshold:.4f}, Target: {target}, Best: {best_metrics}")
        
        return best_metrics, best_threshold
    
    @staticmethod
    def valid_search(
        valid_score: np.ndarray,
        score: np.ndarray, 
        label: np.ndarray, 
        start: float, 
        end: Optional[float] = None, 
        interval: float = 0.1, 
        display_freq: int = DEFAULT_DISPLAY_FREQ, 
        K: int = 0, 
        verbose: bool = True
    ) -> Tuple[Tuple[float, ...], float]:
        """
        Find best F1 score using validation-based threshold search.
        
        Args:
            valid_score: Validation set scores
            score: Test set scores
            label: Test set labels
            start: Start threshold
            end: End threshold
            interval: Search interval
            display_freq: Frequency of progress display
            K: Percentile parameter
            verbose: Whether to show progress
            
        Returns:
            Tuple of (best_metrics, best_threshold)
        """
        if end is None:
            end = start
            interval = 1.0
        
        search_interval = interval
        search_range = end - start
        search_lower_bound = start
        
        if verbose:
            logger.info(f"Search range: {search_lower_bound} to {search_lower_bound + search_range}")
        
        best_metrics = (-1.0, -1.0, -1.0)
        best_threshold = 0.0

        for i in range(int(search_range // search_interval)):
            threshold = np.percentile(valid_score, 100 - (i + 1) * search_interval)
            target = SequenceMetrics.calc_seq(score, label, threshold, K=K, calc_latency=True)
            
            if target[0] > best_metrics[0]:
                best_threshold = threshold
                best_metrics = target
            
            if verbose and i % display_freq == 0:
                logger.info(f"Current threshold: {threshold:.4f}, Target: {target}, Best: {best_metrics}")
        
        return best_metrics, best_threshold


class SequenceMetrics:
    """Sequence-based evaluation metrics for anomaly detection."""
    
    @staticmethod
    def calc_seq(
        score: np.ndarray, 
        label: np.ndarray, 
        threshold: float, 
        K: int = 0, 
        calc_latency: bool = False
    ) -> Union[Tuple[float, ...], Tuple[float, ...]]:
        """
        Calculate sequence-based metrics for anomaly detection.
        
        Args:
            score: Anomaly scores
            label: Ground truth labels
            threshold: Detection threshold
            K: Percentile parameter
            calc_latency: Whether to calculate latency
            
        Returns:
            Tuple of metrics (F1, precision, recall, TP, TN, FP, FN, ROC_AUC, AUPRC, [latency])
        """
        roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        
        if calc_latency:
            predict, latency = ThresholdOptimization.pa_percentile(
                score, label, threshold, K=K, calc_latency=True
            )
            point_metrics = PointMetrics.calc_point2point(predict, label)
            return point_metrics + (roc_auc, auprc, latency)
        else:
            predict = ThresholdOptimization.pa_percentile(score, label, threshold, K=K)
            point_metrics = PointMetrics.calc_point2point(predict, label)
            return point_metrics + (roc_auc, auprc)
    
    @staticmethod
    def get_best_f1(score: np.ndarray, label: np.ndarray) -> Tuple[Tuple[float, ...], float]:
        """
        Find best F1 score using refined search method.
        
        Args:
            score: Anomaly scores
            label: Ground truth labels
            
        Returns:
            Tuple of (best_metrics, best_threshold)
        """
        if score.shape != label.shape:
            raise ValueError("score and label must have the same shape")
        
        logger.info("Computing best F1 score...")
        
        # Count total anomalies
        total_anomalies = np.sum(label > DEFAULT_ANOMALY_LABEL_THRESHOLD)
        
        # Build search set
        search_set = []
        flag = 0
        cur_anomaly_len = 0
        cur_min_anomaly_score = float('inf')
        
        for i in range(label.shape[0]):
            if label[i] > DEFAULT_ANOMALY_LABEL_THRESHOLD:
                if flag == 1:
                    cur_anomaly_len += 1
                    cur_min_anomaly_score = min(score[i], cur_min_anomaly_score)
                else:
                    flag = 1
                    cur_anomaly_len = 1
                    cur_min_anomaly_score = score[i]
            else:
                if flag == 1:
                    flag = 0
                    search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                    search_set.append((score[i], 1, False))
                else:
                    search_set.append((score[i], 1, False))
        
        if flag == 1:
            search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
        
        search_set.sort(key=lambda x: x[0])
        
        # Find best F1
        best_f1 = -1.0
        best_threshold = 1.0
        P = 0
        TP = 0
        
        for score_val, length, is_anomaly in search_set:
            P += length
            if is_anomaly:
                TP += length
            
            precision = TP / (P + EPSILON)
            recall = TP / (total_anomalies + EPSILON)
            f1 = 2 * precision * recall / (precision + recall + EPSILON)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = score_val
        
        logger.info(f"Best F1: {best_f1:.4f}, Threshold: {best_threshold:.4f}")
        
        best_precision = TP / (P + EPSILON)
        best_recall = TP / (total_anomalies + EPSILON)
        
        return (
            best_f1, best_precision, best_recall, TP,
            score.shape[0] - P - total_anomalies + TP,
            P - TP, total_anomalies - TP
        ), best_threshold
    
    @staticmethod
    def get_adjusted_composite_metrics(
        score: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate composite metrics under point-adjust approach.
        
        Args:
            score: Anomaly scores (higher = more anomalous)
            label: Ground truth labels
            
        Returns:
            Tuple of (AUROC, AP, F1, precision, recall, FPR, TPR, thresholds)
        """
        # Convert reconstruction probability to anomaly score
        score = -score
        
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        
        # Adjust scores for segment detection
        splits = np.where(label[1:] != label[:-1])[0] + 1
        is_anomaly = label[0] == 1
        pos = 0
        
        for split in splits:
            if is_anomaly:
                score[pos:split] = np.max(score[pos:split])
            is_anomaly = not is_anomaly
            pos = split
        
        # Handle last segment
        if is_anomaly:
            score[pos:] = np.max(score[pos:])
        
        # Calculate metrics
        fpr, tpr, thresholds = roc_curve(y_true=label, y_score=score, drop_intermediate=False)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true=label, probas_pred=score)
        
        # Find best F1
        f1 = np.max(2 * precision * recall / (precision + recall + EPSILON))
        ap = average_precision_score(y_true=label, y_score=score, average=None)
        
        return auroc, ap, f1, precision, recall, fpr, tpr, thresholds


class AdvancedMetrics:
    """Advanced evaluation metrics and methods."""
    
    @staticmethod
    def anomaly_metric(scores: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate anomaly detection metrics using Youden's J statistic.
        
        Args:
            scores: Anomaly scores
            true: Ground truth labels
            
        Returns:
            Tuple of (precision, recall, f1, auroc)
        """
        fpr, tpr, thresholds = roc_curve(true, scores, pos_label=1)
        J = tpr - fpr
        ix = np.argmax(J)
        pred = np.where(scores < thresholds[ix], 0, 1)
        
        precision = precision_score(true, pred, pos_label=1)
        recall = recall_score(true, pred, pos_label=1)
        f1 = f1_score(true, pred, pos_label=1, average='micro')
        auroc = roc_auc_score(true, pred)
        
        return precision, recall, f1, auroc
    
    @staticmethod
    def percentile_search(
        combined_energy: np.ndarray, 
        score: np.ndarray, 
        label: np.ndarray, 
        anomaly_ratio: float
    ) -> Tuple[Tuple[float, ...], float]:
        """
        Search for optimal threshold using percentile-based approach.
        
        Args:
            combined_energy: Combined energy scores
            score: Anomaly scores
            label: Ground truth labels
            anomaly_ratio: Expected anomaly ratio
            
        Returns:
            Tuple of (metrics, threshold)
        """
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
        target = SequenceMetrics.calc_seq(score, label, threshold, calc_latency=True)
        return target, threshold


# Backward compatibility aliases
def RSE(pred, true): return AnomalyMetrics.rse(pred, true)
def CORR(pred, true): return AnomalyMetrics.corr(pred, true)
def MAE(pred, true): return AnomalyMetrics.mae(pred, true)
def MSE(pred, true): return AnomalyMetrics.mse(pred, true)
def RMSE(pred, true): return AnomalyMetrics.rmse(pred, true)
def MAPE(pred, true): return AnomalyMetrics.mape(pred, true)
def MSPE(pred, true): return AnomalyMetrics.mspe(pred, true)
def metric(pred, true): return AnomalyMetrics.calculate_all_metrics(pred, true)
def calc_point2point(predict, actual): return PointMetrics.calc_point2point(predict, actual)
def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False): 
    return ThresholdOptimization.adjust_predicts(score, label, threshold, pred, calc_latency)
def PA_percentile(score, label, threshold=None, pred=None, K=100, calc_latency=False):
    return ThresholdOptimization.pa_percentile(score, label, threshold, pred, K, calc_latency)
def calc_seq(score, label, threshold, K=0, calc_latency=False):
    return SequenceMetrics.calc_seq(score, label, threshold, K, calc_latency)
def bf_search(score, label, start, end=None, step_num=1, display_freq=1, K=0, verbose=True):
    return ThresholdOptimization.bf_search(score, label, start, end, step_num, display_freq, K, verbose)
def valid_search(valid_score, score, label, start, end=None, interval=0.1, display_freq=1, K=0, verbose=True):
    return ThresholdOptimization.valid_search(valid_score, score, label, start, end, interval, display_freq, K, verbose)
def get_best_f1(score, label): return SequenceMetrics.get_best_f1(score, label)
def get_adjusted_composite_metrics(score, label): return SequenceMetrics.get_adjusted_composite_metrics(score, label)
def anomaly_metric(scores, true): return AdvancedMetrics.anomaly_metric(scores, true)
def percentile_search(combined_energy, score, label, anomaly_ratio): 
    return AdvancedMetrics.percentile_search(combined_energy, score, label, anomaly_ratio)