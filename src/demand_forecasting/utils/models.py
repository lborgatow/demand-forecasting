from typing import Dict, List, Tuple, Union, Any, Type
from functools import partial

import polars as pl
import numpy as np
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import ARIMA

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


def reverse_transformations(y_true: np.ndarray, y_pred: np.ndarray, original_data: np.ndarray, 
                            original_data_idx: int, transformation: str) -> Tuple[np.ndarray]:
    """Reverse the transformation of test data and predictions.

    Args:
        y_true (np.ndarray): Test data.
        y_pred (np.ndarray): Predict data.
        original_data (np.ndarray): Original data.
        original_data_idx (int): Initial index of original data for data reversals that are not predictions.
        transformation (str): Transformation used.

    Returns:
        Tuple[np.ndarray]: Test data and predictions with the transformation reversed.
    """
    
    test = reverse_transformation(original_data=original_data, transformed_data=y_true, 
                                  transformation=transformation, original_data_idx=original_data_idx, 
                                  is_predict=False)
    
    preds = reverse_transformation(original_data=original_data, transformed_data=y_pred, 
                                  transformation=transformation, original_data_idx=original_data_idx, 
                                  is_predict=True)
    
    return test, preds
    

# =============================================== Simple Moving Average ===============================================

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
    test, preds = reverse_transformations(y_true=test, y_pred=preds, original_data=original_data["y"].to_numpy(),
                                          original_data_idx=test_idx[0], transformation=transformation)
    preds = treat_predictions(preds=preds)
    
    naive = repeat_val(val=original_data["y"].to_numpy()[train_idx[-1]], size=test_size)

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

    data_array = preparated_data_dict.get("transformed_data")["y"].to_numpy()
    cv = data_separators_dict.get("cv")
    windows = parameters.get("SEASONALITIES") 
    
    metrics, key_metric = parameters.get("METRICS"), parameters.get("KEY_METRIC")
    metrics_means = {}

    for window in windows:
        metrics_per_window = np.array([fit_cv_sma(global_data, preparated_data_dict, parameters, \
            window, train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)])
        aux = np.mean(metrics_per_window, axis=0)
        metrics_means[window] = {metrics[i]: aux[i] for i in range(len(metrics))}

    best_window = min(metrics_means, key=lambda w: metrics_means[w][key_metric])
    best_metrics = list(metrics_means[best_window].values())
    
    return {
        "best_window": best_window,
        "best_metrics": best_metrics
    }

# ======================================================= Sktime ======================================================

def fit_cv_sktime(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                  data_separators_dict: Dict[str, Union[List[int], Type[EWS]]],
                  model: Any, model_params: Dict[str, Any], 
                  train_idx: np.ndarray, test_idx: np.ndarray) -> List[float]:
    """Get performance metrics for Sktime library models using cross-validation.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        model (Any): Model used for testing and predictions.
        model_params (Dict[str, Any]): Dictionary with the model training parameters and their respective values.
        train_idx (np.ndarray): Array with training data indices.
        test_idx (np.ndarray): Array with test data indexes.

    Returns:
        List[float]: List with the results of the calculated metrics.
    """
    
    data = preparated_data_dict.get("transformed_data").to_pandas()
    series = data.set_index("ds").sort_index().asfreq("D")["y"].squeeze()
    fh_test = data_separators_dict.get("fh_test")
    
    train = series.iloc[train_idx]
    test = series.iloc[test_idx].values

    model_fit = model(**model_params)
    model_fit.fit(train)
    preds = np.array(model_fit.predict(fh=fh_test)).ravel()

    original_data, transformation = preparated_data_dict.get("original_data"), preparated_data_dict.get("transformation")
    test, preds = reverse_transformations(y_true=test, y_pred=preds, original_data=original_data["y"].to_numpy(),
                                          original_data_idx=test_idx[0], transformation=transformation)
    preds = treat_predictions(preds=preds)
    
    naive = repeat_val(val=original_data["y"].to_numpy()[train_idx[-1]], size=len(fh_test))
    
    return global_data.get_metrics(y_true=test, y_pred=preds, y_naive=naive)

# ==================================== ExponentialSmoothing (Sktime - statsmodels) ====================================

def objective_expsmoothing(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                           data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the ExponentialSmoothing model from the Sktime library.

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

    sp = parameters.get("SEASONALITIES")

    model_params = {
        "random_state": 42,
        "use_brute": False,
        "sp": trial.suggest_categorical("sp", sp)
    }
    if min(data_array) <= 0.0:
        model_params["trend"] = trial.suggest_categorical("trend_min<=0", ["additive", None])
        model_params["seasonal"] = trial.suggest_categorical("seasonal_min<=0", ["additive", None])
    else:
        model_params["trend"] = trial.suggest_categorical("trend_min>0", ["additive", "multiplicative", None])
        model_params["seasonal"] = trial.suggest_categorical("seasonal_min>0", ["additive", "multiplicative", None])

    fit_cv_partial = partial(fit_cv_sktime, global_data, preparated_data_dict, data_separators_dict, ExponentialSmoothing, model_params)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)

# ============================================= ARIMA (Sktime - pmdarima) =============================================

def objective_arima(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                    data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the ARIMA model from the Sktime library.

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

    p = trial.suggest_int("p", 1, 4)
    d = trial.suggest_int("d", 0, 1)
    q = trial.suggest_int("q", 0, 4)

    model_params = {
        "suppress_warnings": True,
        "maxiter": 1,
        "order": (p, d, q),
        "method": trial.suggest_categorical("method", ["nm", "lbfgs", "powell"])
    }

    fit_cv_partial = partial(fit_cv_sktime, global_data, preparated_data_dict, data_separators_dict, ARIMA, model_params)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
    
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)
