from typing import Dict, List, Union, Any, Type

import polars as pl
import numpy as np
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS

from .global_data import GlobalData
from .data_preparation import reverse_transformation


def repeat_val(val: float, size: int) -> np.ndarray:
    """Create an array with repeated values.

    Args:
        val (float): Value to be repeated.
        size (int): Size of the array.

    Returns:
        np.ndarray: Array with the value repeated "size" times.
    """
    
    return np.full(size, val, np.float64)


def treat_predictions(preds: np.ndarray) -> np.ndarray:
    """Treat model predictions.

    Args:
        preds (np.ndarray): Model predictions.

    Returns:
        np.ndarray: Predictions treated.
     """
    
    return np.nan_to_num(np.maximum(np.round(preds), 0), nan=0.0)

# ====================================== Simple Moving Average ======================================

def fit_cv_sma(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
               parameters: Dict[str, Any], window: int, train_idx: np.ndarray, test_idx: np.ndarray) -> List[float]:
    """Get performance metrics for a Simple Moving Average model using cross-validation.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        parameters (Dict[str, Any]): Dictionary with global parameters.
        window (int): Amount of data used in the moving average window.
        train_idx (np.ndarray): Array with training data indices.
        test_idx (np.ndarray): Array with test data indexes.
    Returns:
        List[float]: List with the results of the calculated metrics.
    """ 
    
    data = preparated_data_dict.get("transformed_data")["y"].to_numpy()
    test_size = parameters.get('TEST_SIZE')

    train = data[train_idx]
    test = data[test_idx]

    model = train[-window:].mean()
    preds = repeat_val(val=model, size=test_size)
    
    original_data, transformation = preparated_data_dict.get("original_data"), preparated_data_dict.get("transformation")
    preds = reverse_transformation(original_data=original_data, transformed_data=preds, 
                                   transformation=transformation, is_predict=True)
    
    preds = treat_predictions(preds=preds)

    naive = repeat_val(val=train[-1], size=test_size)
    
    return global_data.get_metrics(y_true=test, y_pred=preds, y_naive=naive)


def get_results_cv_sma(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                       data_separators_dict: Dict[str, Union[List[int], Type[EWS]]],
                       parameters: Dict[str, Any]) -> List[float]:
    """Get performance metrics from cross-validation of the Simple Moving Average model.
    
    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        List[float]: List of performance metrics for the best model window.
    """

    data = preparated_data_dict.get("transformed_data")["y"].to_numpy()
    cv = data_separators_dict.get("cv")
    windows = parameters.get('SMA_WINDOWS') 
    
    metrics, key_metric = parameters.get("METRICS"), parameters.get("KEY_METRIC")
    metrics_means = {}

    for window in windows:
        metrics_per_window = np.array([fit_cv_sma(global_data, preparated_data_dict, parameters, \
            window, train_idx, test_idx) for train_idx, test_idx in cv.split(data)])
        aux = np.mean(metrics_per_window, axis=0)
        metrics_means[window] = {metrics[i]: aux[i] for i in range(len(metrics))}

    best_window_idx = min(metrics_means, key=lambda w: metrics_means[w][key_metric])
    
    # best_window = windows[best_window_idx]
    best_metrics = list(metrics_means[best_window_idx].values())
    
    return best_metrics
