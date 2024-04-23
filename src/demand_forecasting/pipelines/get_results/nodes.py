from typing import Dict, List, Any

import polars as pl
from joblib import Parallel, delayed

from demand_forecasting.utils.global_data import GlobalData
from demand_forecasting.utils.data_preparation import prepare_data
from demand_forecasting.utils.data_separation import data_separators
from demand_forecasting.utils.get_results import get_models_results


def run_predictions(global_data: GlobalData, unique_id: str, parameters: Dict[str, Any]) -> Any:
    """Run and concatenate model predictions for the combination.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_id (str): Unique ID (store_item).
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        Any
    """

    preparated_data_dict = prepare_data(global_data=global_data, unique_id=unique_id)
    data_separators_dict = data_separators(preparated_data_dict=preparated_data_dict, parameters=parameters)
    
    return get_models_results(global_data=global_data, preparated_data_dict=preparated_data_dict, 
                              data_separators_dict=data_separators_dict, parameters=parameters)


def run_parallel_predictions(global_data: GlobalData, unique_ids: List[str], parameters: Dict[str, Any]) -> Any:
    """Run models predictions in parallel.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_ids (List[str]): List with unique ids. 
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        Any
    """

    return Parallel(n_jobs=-1)(delayed(run_predictions)(global_data, unique_id, parameters) \
        for unique_id in unique_ids)