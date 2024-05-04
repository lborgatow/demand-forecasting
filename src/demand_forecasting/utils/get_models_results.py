import time
from typing import Dict, List, Type, Union, Callable, Any
from functools import partial

import polars as pl
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS

from .global_data import GlobalData
from .models import get_results_cv_sma
from .models_validation import get_optuna_study, get_best_trial_results, get_results_df


def get_sma_results(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                    data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], parameters: Dict[str, Any]) -> pl.DataFrame:
    """Get the results of the Simple Moving Average model.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the results for the Simple Moving Average model.
    """

    start = time.time()

    results_cv_sma = get_results_cv_sma(global_data=global_data, preparated_data_dict=preparated_data_dict,
                                        data_separators_dict=data_separators_dict, parameters=parameters)
    sma_metrics = results_cv_sma.get("best_metrics")

    end = time.time()
    execution_time = end - start

    unique_id = preparated_data_dict.get("original_data")["unique_id"][0]
    transformation = preparated_data_dict.get("transformation")

    return get_results_df(unique_id=unique_id, transformation=transformation, model="SimpleMovingAverage", 
                          execution_time=execution_time, metrics_values=sma_metrics, parameters=parameters)


def get_sktime_results(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                       data_separators_dict: Dict[str, Union[List[int], Type[EWS]]], model: str, model_objective: Callable, 
                       parameters: Dict[str, Any]) -> pl.DataFrame:
    """Get model results from the Sktime library.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        model (str): Name of the model used for testing and predictions.
        model_objective (Callable): Optuna objective function to optimize model hyperparameters.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the results for the Sktime library model.
    """

    start = time.time()

    objective = partial(model_objective, global_data=global_data, preparated_data_dict=preparated_data_dict,
                        data_separators_dict=data_separators_dict, parameters=parameters)
    
    n_trials = parameters.get("OPTUNA_TRIALS")
    study = get_optuna_study(objective_func=objective, trials=n_trials, len_metrics=len(global_data.metrics))

    best_trial_results = get_best_trial_results(study=study, parameters=parameters)
    metrics = best_trial_results.get("trial_metrics")

    end = time.time()
    execution_time = end - start

    unique_id = preparated_data_dict.get("original_data")["unique_id"][0]
    transformation = preparated_data_dict.get("transformation")

    return get_results_df(unique_id=unique_id, transformation=transformation, model=model, 
                          execution_time=execution_time, metrics_values=metrics, 
                          parameters=parameters)