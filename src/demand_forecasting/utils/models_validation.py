from typing import Dict, List, Union, Callable, Any, Type

import polars as pl
import numpy as np
import optuna

# import warnings

# optuna.logging.set_verbosity(optuna.logging.CRITICAL)
# warnings.filterwarnings(action="ignore")


def get_optuna_study(objective_func: Callable, trials: int, len_metrics: int) -> Type[optuna.Study]:
    """Get a study of optuna optimization.

    Args:
        objective (Callable): Objective function.
        trials (int): Number of experiments in the study.
        len_metrics (int): Number of evaluation metrics.

    Returns:
        Type[optuna.Study]: Study of optuna optimization.
    """

    study = optuna.create_study(
        directions=['minimize'] * len_metrics, 
        sampler=optuna.samplers.TPESampler(
            seed=42, warn_independent_sampling=False
        )
    )
    study.optimize(objective_func, n_trials=trials)
    
    return study


def get_optuna_metrics(metrics_cv: List[float], len_metrics: int) -> List[float]:
    """Get performance metrics during optuna experiments.

    Args:
        metrics_cv (List[float]): List of metrics obtained from training and prediction
        with cross-validation.
        len_metrics (int): Number of evaluation metrics.

    Returns:
        List[float]: List with the averages of each metric.
    """
    
    return [np.array([x[i] for x in metrics_cv]).mean() for i in range(len_metrics)]


def get_best_trial_results(study: Type[optuna.Study], parameters: Dict[str, Any]) -> Dict[str, Union[Dict[str, Any], List[float]]]:
    """Get the parameters and metrics of the best experiment from the optuna study.

    Args:
        study (Type[optuna.Study]): Study of optuna optimization.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        Dict[str, Union[Dict[str, Any], List[float]]]: Dictionary with parameters and performance 
        metrics of the best experiment in the optuna study.
    """
    
    metrics = parameters.get("METRICS")
    key_metric = parameters.get("KEY_METRIC")
    key_metric_idx = metrics.index(key_metric)
    
    best_trials = study.best_trials
    trial_with_lowest_metric = min(best_trials, key=lambda t: t.values[key_metric_idx])
        
    params = trial_with_lowest_metric.params
    trial_metrics = trial_with_lowest_metric.values
        
    return {
        "best_params": params,
        "trial_metrics": trial_metrics
    }


def get_results_df(unique_id: str, transformation: str, model: str, execution_time: float, 
                   metrics_values: List[float], parameters: Dict[str, Any]) -> pl.DataFrame:
    """Get the DataFrame with model results.

    Args:
        unique_id (str): Unique ID.
        transformation (str): Transformation to try to make the series stationary.
        model (str): Model name.
        execution_time (str): Model execution time.
        metrics_values (List[float]): List of model metrics in order.
        parameters (Dict[str, Any): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the model's results, metrics and predictions.
    """

    store, item = unique_id.split("_")
    metrics = parameters.get("METRICS")

    results = {
        "unique_id": [unique_id], 
        "store": [store], 
        "item": [item], 
        "transformation": [transformation], 
        "model": [model], 
        "execution_time": [execution_time],
    }

    for i, metric in enumerate(metrics):
        results[metric] = [metrics_values[i]]
    
    return pl.DataFrame(results)