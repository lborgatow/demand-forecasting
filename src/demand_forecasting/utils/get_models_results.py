import time
from typing import Dict, List, Type, Union, Any

import polars as pl
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS

from .global_data import GlobalData
from .models import get_results_cv_sma
from .models_validation import get_results_df


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

    sma_metrics = get_results_cv_sma(global_data=global_data, preparated_data_dict=preparated_data_dict,
                                     data_separators_dict=data_separators_dict, parameters=parameters)
    
    end = time.time()
    execution_time = end - start

    uid = preparated_data_dict.get("original_data")["uid"][0]

    return get_results_df(uid=uid, model="SimpleMovingAverage", execution_time=execution_time,
                          metrics_values=sma_metrics, parameters=parameters)

