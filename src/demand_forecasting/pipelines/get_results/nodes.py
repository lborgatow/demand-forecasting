from typing import Dict, List, Any
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from time import time

import polars as pl
from joblib import Parallel, delayed
from humanfriendly import format_timespan

from demand_forecasting.utils.global_data import GlobalData
from demand_forecasting.utils.data_preparation import prepare_data
from demand_forecasting.utils.data_separation import data_separators
from demand_forecasting.utils.get_results import get_models_results

manager = Manager()


def run_predictions(global_data: GlobalData, unique_id: str, aux_dict: DictProxy, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Run and concatenate model predictions for the combination.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_id (str): Unique ID (store_item).
        aux_dict (DictProxy): Dictionary to assist in controlling executions.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the results of the models for the unique_id executed.
    """
    
    preparated_data_dict = prepare_data(global_data=global_data, unique_id=unique_id)
    data_separators_dict = data_separators(preparated_data_dict=preparated_data_dict, parameters=parameters)
    
    models_results = get_models_results(global_data=global_data, preparated_data_dict=preparated_data_dict, 
                                        data_separators_dict=data_separators_dict, parameters=parameters)
    
    aux_dict["count"] -= 1
    store, item = unique_id.split("_")
    print(f"STORE: {store}; ITEM: {item}; UID: {unique_id} - CONCLUIDO; {aux_dict.get('count')} restante(s)")
    
    return models_results


def run_parallel_predictions(global_data: GlobalData, unique_ids: List[str], parameters: Dict[str, Any]) -> pl.DataFrame:
    """Run models predictions in parallel.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_ids (List[str]): List with unique ids. 
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with concatenated results.
    """
    
    print_title = lambda title, size: print("\n" + ("#" * size) + "\n" + title.center(size) + "\n" + ("#" * size) + "\n")
    
    start = time()
    
    print_title(title="INICIADO", size=100)
    
    aux_dict = manager.dict()
    aux_dict["count"] = len(unique_ids)
    print(f"UIDs encontrados: {aux_dict.get('count')}\n")
    
    results = Parallel(n_jobs=-1)(delayed(run_predictions)(global_data, unique_id, aux_dict, parameters) \
        for unique_id in unique_ids)
    concatenated_results = pl.concat(results).sort(by="unique_id")
    
    end = time()
    total_time = end - start
    print(f"\nTempo total: {format_timespan(total_time, max_units=4)}")
    
    print_title(title="FINALIZADO", size=100)
    
    return concatenated_results