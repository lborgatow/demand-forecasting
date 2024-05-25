from typing import Dict, List, Tuple, Union, Any, Type
from functools import partial

import polars as pl
import numpy as np
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.fbprophet import Prophet
from xgboost import XGBRegressor
import pytimetk as tk
from darts.timeseries import TimeSeries
from darts.utils.utils import ModelMode, TrendMode, SeasonalityMode
from darts.models import FourTheta, NHiTSModel, TiDEModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

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
    test_size = parameters.get("TEST_SIZE")

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

# ================================================= Prophet (Sktime) ==================================================

def objective_prophet(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                      data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the Prophet model from the Sktime library.

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
        "freq": "D",
        "add_country_holidays": {"country_name": "Brazil"},
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
        "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10, log=True),
        "changepoint_range": trial.suggest_float("changepoint_range", 0.8, 0.95)
    }
    if min(data_array) <= 0.0:
        model_params["seasonality_mode"] = trial.suggest_categorical("seasonality_mode_min<=0", ["additive"])
    else:
        model_params["seasonality_mode"] = trial.suggest_categorical("seasonality_mode_min>0", ["additive", "multiplicative"])

    fit_cv_partial = partial(fit_cv_sktime, global_data, preparated_data_dict, data_separators_dict, Prophet, model_params)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)

# ================================================= XGBoost (xgboost) ==================================================

def create_features_pytimetk(df: pl.DataFrame, future: int, lags: int) -> Tuple[pl.DataFrame]:
    """Create features from a series to perform predictions with tree models like XGBoost.

    Args:
        df (pl.DataFrame): Series to predict transformed into DataFrame.
        future (int): Number of future dates for prediction.
        lags (int): Lags for generating model features.

    Returns:
        Tuple[pl.DataFrame]: Tuple with the DataFrame for training/testing and the DataFrame for 
        future predictions.
    """
    
    df = df.select(["ds", "y"]).sort(by="ds").to_pandas()
    future_df = df.future_frame("ds", future)
    df_dates = future_df.augment_timeseries_signature(date_column="ds")
    df_lags = df_dates.augment_lags(
        date_column="ds", 
        value_column="y", 
        lags=[lag for lag in range(30, lags+1, 30)]
    )
        
    lag_columns = [col for col in df_lags.columns if "lag" in col]

    df_no_nas = df_lags.dropna(subset=lag_columns, inplace=False).set_index("ds")
    df_no_nas.index = df_no_nas.index.to_period()
    
    columns = [ 
        "y"
        , "ds_year"
        , "ds_half"
        , "ds_quarter"
        , "ds_month"
        # , "ds_yweek"
        , "ds_mweek"
        , "ds_wday"
        , "ds_mday"
    ] + [f"y_lag_{lag}" for lag in range(30, lags+1, 30)] 
    
    final_df = pl.DataFrame(df_no_nas[df_no_nas["y"].notnull()][columns])
    future_df = pl.DataFrame(df_no_nas[df_no_nas["y"].isnull()][columns])
    
    return final_df, future_df


def fit_cv_xgboost(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]], train_df: pl.DataFrame, 
                   model_params: Dict[str, Any], parameters: Dict[str, Any], train_idx: np.ndarray, test_idx: np.ndarray) -> List[float]:
    """Get performance metrics for XGBoost model using cross-validation.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        train_df (pl.DataFrame): DataFrame with the data for XGBoost training.
        model_params (Dict[str, Any]): Dictionary with the model training parameters and their respective values.
        parameters (Dict[str, Any]): Dictionary with global parameters.
        train_idx (np.ndarray): Array with training data indices.
        test_idx (np.ndarray): Array with test data indexes.

    Returns:
        List[float]: List with the results of the calculated metrics.
    """
    
    test_size = parameters.get("TEST_SIZE")
    
    X_train = train_df[train_idx].select(train_df.columns[1:])
    y_train = train_df[train_idx].select(train_df.columns[0])
    X_test = train_df[test_idx].select(train_df.columns[1:])
    y_test = train_df[test_idx].select(train_df.columns[0]).to_numpy().ravel()
    
    X_train = X_train[:-test_size]
    X_val = X_train[-test_size:]
    y_train = y_train[:-test_size]
    y_val = y_train[-test_size:]

    model_fit = XGBRegressor(**model_params)
    
    model_fit.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=False)
    
    preds = model_fit.predict(X_test)
    preds = np.array(model_fit.predict(X_test)).ravel()

    original_data, transformation = preparated_data_dict.get("original_data"), preparated_data_dict.get("transformation")
    test, preds = reverse_transformations(y_true=y_test, y_pred=preds, original_data=original_data["y"].to_numpy(),
                                          original_data_idx=test_idx[0], transformation=transformation)
    preds = treat_predictions(preds=preds)
    
    naive = repeat_val(val=original_data["y"].to_numpy()[train_idx[-1]], size=test_size)
    
    return global_data.get_metrics(y_true=test, y_pred=preds, y_naive=naive)


def objective_xgboost(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]], 
                      parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the XGBBoost model.

    Args:
        trial (Any): Experiment of the optuna study.
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        List[float]: List with the values ​​of the model's performance metrics.
    """
    
    data = preparated_data_dict.get("transformed_data")
    test_size = parameters.get("TEST_SIZE")
    len_metrics = len(global_data.metrics)
    seasonalities = parameters.get("SEASONALITIES")
    
    lags = trial.suggest_categorical("lags", seasonalities)
    
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmsle",
        "random_state": 42,
        "early_stopping_rounds": 20,
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 500, 700, 25),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.2, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.2, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel",0.2, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    }
    
    train_df, _ = create_features_pytimetk(df=data, lags=lags, future=test_size)
    
    folds, step_length = parameters.get("CV_FOLDS"), parameters.get("CV_STEP_LENGTH")
    fh_test = list(range(1, test_size+1))
    initial_window = len(train_df) - (test_size + ((folds - 1)*step_length))
    cv = EWS(fh=fh_test, initial_window=initial_window, step_length=step_length)
    
    fit_cv_partial = partial(fit_cv_xgboost, global_data, preparated_data_dict, train_df, model_params, parameters)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(train_df.to_pandas())]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)

# ======================================================= Darts =======================================================

def fit_cv_darts(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                 model: Any,  is_dl: bool, model_params: Dict[str, Any], parameters: Dict[str, Any],
                 train_idx: np.ndarray, test_idx: np.ndarray) -> List[float]:
    """Get performance metrics for Darts library models using cross-validation.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        model (Any): Model used for testing and predictions.
        is_dl (bool): Bool informing whether or not it is a deep learning model.
        model_params (Dict[str, Any]): Dictionary with the model training parameters and their respective values.
        parameters (Dict[str, Any]): Dictionary with global parameters.
        train_idx (np.ndarray): Array with training data indices.
        test_idx (np.ndarray): Array with test data indexes.

    Returns:
        List[float]: List with the results of the calculated metrics.
    """
    
    data = preparated_data_dict.get("transformed_data").to_pandas()
    series = TimeSeries.from_dataframe(df=data, time_col="ds", value_cols="y", freq="D")
    test_size = parameters.get("TEST_SIZE")
    
    train = series[train_idx.tolist()]
    test = series[test_idx.tolist()].univariate_values()
    
    model_fit = model(**model_params)
    if is_dl:
        val_size = model_params.get("input_chunk_length") + model_params.get("output_chunk_length")
        train = series[:-val_size]
        val = series[-val_size:]
        model_fit.fit(train, val_series=val)
    else:
        model_fit.fit(train)
    
    preds = np.array(model_fit.predict(n=test_size).values()).ravel()

    original_data, transformation = preparated_data_dict.get("original_data"), preparated_data_dict.get("transformation")
    test, preds = reverse_transformations(y_true=test, y_pred=preds, original_data=original_data["y"].to_numpy(),
                                          original_data_idx=test_idx[0], transformation=transformation)
    preds = treat_predictions(preds=preds)
    
    naive = repeat_val(val=original_data["y"].to_numpy()[train_idx[-1]], size=test_size)
    
    return global_data.get_metrics(y_true=test, y_pred=preds, y_naive=naive)

# ================================================= FourTheta (Darts) =================================================

def objective_fourtheta(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                        data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the FourTheta model from the Darts library.

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
    
    mm_add = ModelMode.ADDITIVE
    mm_mult = ModelMode.MULTIPLICATIVE
    tm_lin = TrendMode.LINEAR
    tm_exp = TrendMode.EXPONENTIAL
    sm_none = SeasonalityMode.NONE
    sm_add = SeasonalityMode.ADDITIVE
    sm_mult = SeasonalityMode.MULTIPLICATIVE

    model_params = {
        "theta": trial.suggest_int("theta", 0, 4),
        "seasonality_period": trial.suggest_categorical("seasonality_period", sp),
        "trend_mode": trial.suggest_categorical("trend_mode", [tm_lin, tm_exp])

    }
    if min(data_array) <= 0.0:
        model_params["model_mode"] = trial.suggest_categorical("model_mode_min<=0", [mm_add])
        model_params["season_mode"] = trial.suggest_categorical("season_mode_min<=0", [sm_none, sm_add])
    else:
        model_params["model_mode"] = trial.suggest_categorical("model_mode_min>0", [mm_add, mm_mult])
        model_params["season_mode"] = trial.suggest_categorical("season_mode_min>0", [sm_none, sm_add, sm_mult])

    fit_cv_partial = partial(fit_cv_darts, global_data, preparated_data_dict, FourTheta, False, model_params, parameters)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)

# ================================================== N-HiTS (Darts) ===================================================

def objective_nhits(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                    data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the N-HiTS model from the Darts library.

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
    
    torch.manual_seed(42)

    early_stopper = EarlyStopping("val_loss", min_delta=0.05, patience=3, verbose=False)
    pl_trainer_kwargs = {
        "callbacks": [early_stopper],
        "enable_progress_bar": False,
        "enable_model_summary": False
    }
    
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    
    model_params = {
        "random_state": 42,
        "n_epochs": trial.suggest_int("n_epochs", 20, 50),
        "input_chunk_length": trial.suggest_categorical("input_chunck_length", sp),
        "output_chunk_length": parameters.get("TEST_SIZE"),
        "num_stacks": trial.suggest_int("num_stacks", 1, 3),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.01),
        "layer_widths": trial.suggest_categorical("layer_widths", [64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "loss_fn": torch.nn.MSELoss(),
        "optimizer_kwargs": {"lr": lr},
        "pl_trainer_kwargs": pl_trainer_kwargs
    }
    
    fit_cv_partial = partial(fit_cv_darts, global_data, preparated_data_dict, NHiTSModel, True, model_params, parameters)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)

# ================================================== TiDE (Darts) ===================================================

def objective_tide(trial: Any, global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                   data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> List[float]:
    """Optuna study objective function for the TiDE model from the Darts library.

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
    
    torch.manual_seed(42)

    early_stopper = EarlyStopping("val_loss", min_delta=0.05, patience=3, verbose=False)
    pl_trainer_kwargs = {
        "callbacks": [early_stopper],
        "enable_progress_bar": False,
        "enable_model_summary": False
    }
    
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    
    model_params = {
        "random_state": 42,
        "n_epochs": trial.suggest_int("n_epochs", 20, 50),
        "input_chunk_length": trial.suggest_categorical("input_chunck_length", sp),
        "output_chunk_length": parameters.get("TEST_SIZE"),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 3),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 3),
        "decoder_output_dim": trial.suggest_categorical("decoder_output_dim", [8, 16, 32]),
        "temporal_width_past": 0,
        "temporal_width_future": 0,
        "use_layer_norm": trial.suggest_categorical("use_layer_norm", [True, False]),
        "use_reversible_instance_norm": True,
        "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.01),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "loss_fn": torch.nn.MSELoss(),
        "optimizer_kwargs": {"lr": lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
    }
    
    fit_cv_partial = partial(fit_cv_darts, global_data, preparated_data_dict, TiDEModel, True, model_params, parameters)
    metrics_cv = [fit_cv_partial(train_idx, test_idx) for train_idx, test_idx in cv.split(data_array)]
        
    return get_optuna_metrics(metrics_cv=metrics_cv, len_metrics=len_metrics)