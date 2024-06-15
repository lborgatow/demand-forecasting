from typing import List, Union
from dataclasses import dataclass

import polars as pl
import numpy as np


@dataclass
class Metrics:
    """Class to define and obtain performance metrics."""


    def smape(self, y_true: Union[List[float], np.ndarray, pl.Series], 
              y_pred: Union[List[float], np.ndarray, pl.Series]) -> float:
        """Calculates the Symmetric Mean Absolute Percentage Error (sMAPE) for a forecast.

        Args:
            y_true (Union[List[float], np.ndarray, pl.Series]): True values.
            y_pred (Union[List[float], np.ndarray, pl.Series]): Predicted values.

        Returns:
            float: sMAPE value.
        """
        
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        metric = 2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))
        aux = np.where((y_true == 0) & (y_pred == 0), 0.0, metric)

        return np.mean(aux)


    def rmsle(self, y_true: Union[List[float], np.ndarray, pl.Series], 
             y_pred: Union[List[float], np.ndarray, pl.Series]) -> float:
        """Calculates the Root Mean Squared Log Error (RMSLE) for a forecast.

        Args:
            y_true (Union[List[float], np.ndarray, pl.Series]): True values.
            y_pred (Union[List[float], np.ndarray, pl.Series]): Predicted values.

        Returns:
            float: RMSLE value.
        """
        
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        metric = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
        aux = np.where((y_true == 0) & (y_pred == 0), 0.0, metric)
        
        return np.sqrt(np.mean(aux))
    
    
    def owa(self, y_true: Union[List[float], np.ndarray, pl.Series], 
            y_pred: Union[List[float], np.ndarray, pl.Series],
            y_naive: Union[List[float], np.ndarray, pl.Series]) -> float:
        """Calculates the Overall Weighted Average (RMSLE) for a forecast.

        Args:
            y_true (Union[List[float], np.ndarray, pl.Series]): True values.
            y_pred (Union[List[float], np.ndarray, pl.Series]): Predicted values.
            y_naive (Union[List[float], np.ndarray, pl.Series]): Values predicted by the naive model.

        Returns:
            float: OWA value.
        """
        
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        y_naive = np.array(y_naive).ravel()
        
        model_smape = self.smape(y_true, y_pred)
        naive_smape = self.smape(y_true, y_naive)
        
        model_rmsle = self.rmsle(y_true, y_pred)
        naive_rmsle = self.rmsle(y_true, y_naive)
        
        if (naive_smape == 0.0) and (naive_rmsle == 0.0):
            naive_smape += 1e-10
            naive_rmsle += 1e-10
        
        return ((model_smape / naive_smape) + (model_rmsle / naive_rmsle)) / 2


    def run_metrics(self, metrics: List[str], 
                    y_true: Union[List[float], np.ndarray, pl.Series], 
                    y_pred: Union[List[float], np.ndarray, pl.Series], 
                    y_naive: Union[List[float], np.ndarray, pl.Series]) -> List[float]:
        """Get performance metrics results.

        Arguments:
            metrics (List[str]): List of metrics being calculated.
            y_true (Union[List[float], np.ndarray, pl.Series]): True values.
            y_pred (Union[List[float], np.ndarray, pl.Series]): Predicted values.
            y_naive (Union[List[float], np.ndarray, pl.Series]): Values predicted by the naive model.

        Returns:
            List[float]: List of metrics in order.
        """

        return [
            self.__getattribute__(metric)(y_true, y_pred) if metric != "owa" 
            else self.__getattribute__(metric)(y_true, y_pred, y_naive) for metric in metrics
        ]