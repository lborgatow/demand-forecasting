from typing import Dict, List, Tuple, Any

import polars as pl

from demand_forecasting.utils.data_preprocessing import process_data, get_unique_ids
from demand_forecasting.utils.global_data import GlobalData


def preprocess_data(data: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Perform data preprocessing.

    Args:
        data (pl.DataFrame): DataFrame with temporal data.

    Returns:
        Tuple[pl.DataFrame, List[str]]: Tuple with the processed DataFrame and a 
        list with unique ids.
    """

    # data = pl.DataFrame(data)
    processed_data = process_data(data=data)
    unique_ids = get_unique_ids(data=processed_data)

    return processed_data, unique_ids


def define_global_data(processed_data: pl.DataFrame, parameters: Dict[str, Any]) -> GlobalData:
    """Define the objects that will be used in all prediction runs.

    Args:
        processed_data (pl.DataFrame): DataFrame with the processed temporal data.
        parameters (Dict[str, Dict[str, Any]]): Dictionary with global parameters.

    Returns:
        GlobalData: Object of the GlobalData class.
    """

    return GlobalData(processed_data, parameters.get("METRICS"))