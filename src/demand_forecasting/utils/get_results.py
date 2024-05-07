from typing import Dict, List, Union, Type, Any

import polars as pl
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS

from .global_data import GlobalData
from .models import objective_expsmoothing, objective_arima, objective_prophet
from .get_models_results import get_sma_results, get_sktime_results


def get_models_results(global_data: GlobalData, preparated_data_dict: Dict[str, Union[str, pl.DataFrame]],
                       data_separators_dict: Dict[str, Union[List[int], Type[EWS]]],
                       parameters: Dict[str, Any]) -> pl.DataFrame:
    """Get the results of all models.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        preparated_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with the preparated data.
        data_separators_dict (Dict[str, Union[List[int], Type[EWS]]]): Dictionary with data separators.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the results of all models.
    """

    df_sma = get_sma_results(global_data=global_data, preparated_data_dict=preparated_data_dict,
                             data_separators_dict=data_separators_dict, parameters=parameters)
    
    df_expsmoothing = get_sktime_results(global_data=global_data, preparated_data_dict=preparated_data_dict,
                                         data_separators_dict=data_separators_dict, model="ExponentialSmoothing",
                                         model_objective=objective_expsmoothing, parameters=parameters)
    
    df_arima = get_sktime_results(global_data=global_data, preparated_data_dict=preparated_data_dict,
                                  data_separators_dict=data_separators_dict, model="ARIMA",
                                  model_objective=objective_arima, parameters=parameters)
    
    df_prophet = get_sktime_results(global_data=global_data, preparated_data_dict=preparated_data_dict,
                                    data_separators_dict=data_separators_dict, model="Prophet",
                                    model_objective=objective_prophet, parameters=parameters)
    
    return pl.concat([df_sma, df_expsmoothing, df_arima, df_prophet])