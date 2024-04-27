from typing import Dict, List, Union, Any, Type
from functools import partial

import polars as pl
import numpy as np
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.timeseries import TimeSeries
from darts.models import ExponentialSmoothing

from .global_data import GlobalData
from .data_preparation import reverse_transformation
from .models_validation import get_optuna_metrics


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

    train = data[train_idx.tolist()]
    test = data[test_idx.tolist()]

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
                       parameters: Dict[str, Any]) -> Dict[str, Union[int, List[float]]]:
    """Get performance metrics from cross-validation of the Simple Moving Average model.
    
    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        Dict[str, Union[int, List[float]]]: Dictionary with the value of the best moving average window and 
        a list of performance metrics for that window.
    """

    data = preparated_data_dict.get("transformed_data")["y"].to_numpy()
    cv = data_separators_dict.get("cv")
    windows = parameters.get("SEASONALITIES") 
    
    metrics, key_metric = parameters.get("METRICS"), parameters.get("KEY_METRIC")
    metrics_means = {}

    for window in windows:
        metrics_per_window = np.array([fit_cv_sma(global_data, preparated_data_dict, parameters, \
            window, train_idx, test_idx) for train_idx, test_idx in cv.split(data)])
        aux = np.mean(metrics_per_window, axis=0)
        metrics_means[window] = {metrics[i]: aux[i] for i in range(len(metrics))}

    best_window = min(metrics_means, key=lambda w: metrics_means[w][key_metric])
    best_metrics = list(metrics_means[best_window].values())
    
    return {
        "best_window": best_window,
        "best_metrics": best_metrics
    }

# ====================================== Darts ======================================

def fit_cv_darts(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                 model: Any, model_params: Dict[str, Any], parameters: Dict[str, Any], 
                 train_idx: np.ndarray, test_idx: np.ndarray) -> List[float]:
    """Get performance metrics for Darts library models using cross-validation.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        model (Any): Model used for testing and predictions.
        model_params (Dict[str, Any]): Dictionary with the model training parameters and their respective values.
        parameters (Dict[str, Union[List[str], List[int], str, int]]): Dictionary with global parameters.
        train_idx (np.ndarray): Array with training data indices.
        test_idx (np.ndarray): Array with test data indexes.

    Returns:
        List[float]: List with the results of the calculated metrics.
    """
    
    data = preparated_data_dict.get("transformed_data").to_pandas()
    ts_data = TimeSeries.from_dataframe(data, "ds", "y", freq="1d")
    test_size = parameters.get("TEST_SIZE")
    
    train = ts_data[train_idx.tolist()]
    test = ts_data[test_idx.tolist()].values()

    model_fit = model(**model_params)
    model_fit.fit(train)
    preds = np.array(model_fit.predict(test_size).values()).ravel()

    original_data, transformation = preparated_data_dict.get("original_data"), preparated_data_dict.get("transformation")
    preds = reverse_transformation(original_data=original_data, transformed_data=preds, 
                                   transformation=transformation, is_predict=True)
    preds = treat_predictions(preds=preds)

    naive = repeat_val(val=train.univariate_values()[-1], size=test_size)
        
    return global_data.get_metrics(y_true=test, y_pred=preds, y_naive=naive)

# ====================================== ExponentialSmoothing (Darts) ======================================

def objective_expsmoothing(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                           data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the ExponentialSmoothing model from the Darts library.

    Args:
        trial (Any): Experiment of the optuna study.
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        List[float]: List with the values ​​of the model's performance metrics.
    """

    data_array = preparated_data_dict.get("transformed_data")["y"].to_numpy()
    cv = data_separators_dict.get("cv")
    len_metrics = len(global_data.metrics)
    
    mm_none = ModelMode.NONE
    mm_add = ModelMode.ADDITIVE
    mm_mult = ModelMode.MULTIPLICATIVE
    sm_none = SeasonalityMode.NONE
    sm_add = SeasonalityMode.ADDITIVE
    sm_mult = SeasonalityMode.MULTIPLICATIVE

    seasonal_periods = parameters.get("SEASONALITIES")

    # try:
    model_params = {
        "random_state": 42,
        "seasonal_periods": trial.suggest_categorical("seasonal_periods", seasonal_periods),
    }
    if min(data_array) <= 0.0:
        model_params["trend"] = trial.suggest_categorical("trend", [mm_none, mm_add])
        model_params["seasonal"] = trial.suggest_categorical("seasonal", [sm_none, sm_add])
    else:
        model_params['trend'] = trial.suggest_categorical("trend", [mm_none, mm_add, mm_mult])
        model_params["seasonal"] = trial.suggest_categorical("seasonal", [sm_none, sm_add, sm_mult])
    model_params["kwargs"] = {"initialization_method": trial.suggest_categorical("initialization_method", ["estimated", "heuristic"])}

    fit_cv_partial = partial(fit_cv_darts, global_data, preparated_data_dict, ExponentialSmoothing, model_params, parameters)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
    # except:
    #     return [np.inf] * len_metrics
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)
