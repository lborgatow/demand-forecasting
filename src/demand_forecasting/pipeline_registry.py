"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from demand_forecasting.pipelines import data_preprocessing, get_results


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    
    preprocess_data = data_preprocessing.create_pipeline()
    get_all_results = get_results.create_pipeline()
    
    return {
        "__default__": preprocess_data + get_all_results
    }