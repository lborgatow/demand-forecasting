from typing import Dict, List, Any

import polars as pl


def get_results_df(unique_id: str, model: str, execution_time: float, metrics_values: List[float], 
                   parameters: Dict[str, Any]) -> pl.DataFrame:
    """Get the DataFrame with model results.

    Args:
        unique_id (str): Unique ID.
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
        "unique_id": [unique_id], "store": [store], "item": [item], "model": [model], "execution_time": [execution_time],
    }

    for i, metric in enumerate(metrics):
        results[metric] = [metrics_values[i]]
    
    return pl.DataFrame(results)