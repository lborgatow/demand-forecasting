from typing import Dict, Union

import polars as pl
import numpy as np
from statsmodels.tsa.stattools import adfuller

from demand_forecasting.utils.global_data import GlobalData


def check_stationarity(data: np.ndarray) -> bool:
        """Check the stationarity of a series using the Dickey-Fuller test.

        Args:
            data (np.ndarray): Series to be checked.

        Returns:
            bool: Returns "True" if the series is stationary and "False" if it is not.
        """
        
        return adfuller(data)[1] < 0.05


def data_transformation(original_data: pl.DataFrame) -> Dict[str, Union[str, pl.DataFrame]]:
    """Choose the best transformation for a data to make it stationary.

    Args:
        original_data (pl.DataFrame): Original DataFrame.

    Returns:
        Dict[str, Union[pd.Series, bool, str]]: Dictionary with the prepared data.
    """

    transformations = {
        "original": original_data["y"].to_numpy(),
        # "diff": np.diff(original_data["y"]),
        "log": np.log1p(original_data["y"]),
        "cbrt": np.cbrt(original_data["y"])
    }
    
    for name, transformation in transformations.items():
        data = original_data[1:] if name == "diff" else original_data[:]
        if check_stationarity(transformation):
            return {
                "transformation": name,
                "original_data": original_data,
                "transformed_data": data.with_columns(y=transformation)
            }
    
    return {
            "transformation": "original",
            "original_data": original_data,
            "transformed_data": data.with_columns(y=transformations.get("original"))
        }


def reverse_transformation(original_data: np.ndarray, original_data_idx: int,
                           transformed_data: np.ndarray, transformation: str, 
                           is_predict: bool) -> np.ndarray:
    """Reverse the data transformation.

    Args:
        original_data (np.ndarray): Original data.
        original_data_idx (int): Initial index of original data for data reversals that are not predictions.
        transformed_data (np.ndarray): Transformed data.
        transformation (str): Transformation used.
        is_predict (bool): Boolean that indicates whether the transformed data is a prediction
        or not (due to the differentiation technique).

    Returns:
        np.ndarray: Data with the transformation reversed.
    """
    
    if transformation == "diff":
        if is_predict:
            return np.cumsum(np.insert(transformed_data, 0, original_data[-1]))[1:]
        else:
            return np.cumsum(np.insert(transformed_data, 0, original_data[original_data_idx]))[:-1]
    elif transformation == "log":
        return np.expm1(transformed_data)
    elif transformation == "cbrt":
        return transformed_data**3
    else:
        return transformed_data
    

def prepare_data(global_data: GlobalData, 
                 unique_id: str) -> Dict[str, Union[str, pl.DataFrame]]:
    """Prepare data for predictions.

    Args:
        global_data (GlobalData): Object of the GlobalData class.
        unique_id (Tuple[str]): Unique ID (store_item).

    Returns:
        Dict[str, Union[pd.Series, bool, str]]: Dictionary with the prepared data.
    """
    
    filtered_data = global_data.filter_and_sort_data(unique_id=unique_id)

    return data_transformation(original_data=filtered_data)