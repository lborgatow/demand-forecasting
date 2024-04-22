from typing import List, Union
from dataclasses import dataclass

import polars as pl
import numpy as np

from .metrics import Metrics


@dataclass
class GlobalData(Metrics):
    """Class to define the global data that will be used in all executions."""

    data: pl.DataFrame
    metrics: List[str]


    def filter_and_sort_data(self, unique_id: str) -> pl.DataFrame:
        """Filter the DataFrame data according to the unique id and 
        sort data accordind to the date.

        Arguments:
            unique_id (str): store-item unique id.
        
        Returns:
            pl.DataFrame: DataFrame with the filtered data.
        """

        return self.data.filter(pl.col("uid") == unique_id).sort(by="ds")


    def get_metrics(self, y_true: Union[List[float], np.ndarray, pl.Series], 
                    y_pred: Union[List[float], np.ndarray, pl.Series],
                    y_naive: Union[List[float], np.ndarray, pl.Series]) -> List[float]:
        """Get the metrics when executing the "run_metrics" method of the "Metrics" class.

        Arguments:
            y_true (Union[List[float], np.ndarray, pl.Series]): True values.
            y_pred (Union[List[float], np.ndarray, pl.Series]): Predicted values.
            y_naive (Union[List[float], np.ndarray, pl.Series]): Values predicted by the naive model.

        Returns:
            List[float]: List with metric values.
        """

        return self.run_metrics(
            metrics=self.metrics, 
            y_true=y_true, 
            y_pred=y_pred, 
            y_naive=y_naive
        )