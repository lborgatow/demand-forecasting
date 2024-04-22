from typing import Dict, Any

import polars as pl

from demand_forecasting.utils.global_data import GlobalData
from demand_forecasting.utils.data_preparation import prepare_data


def run_predictions(global_data: GlobalData, unique_id: str, 
                    parameters: Dict[str, Any]) -> pl.DataFrame:
    """Run and concatenate model predictions for the combination.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_id (str): Unique ID (store_item).
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with the results of the concatenated models for the combination.
    """

    preparated_data_dict = prepare_data(global_data=global_data, unique_id=unique_id)
    
    pass