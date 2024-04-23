from typing import List, Dict, Union, Type, Any

import polars as pl
from sktime.forecasting.model_selection import ExpandingWindowSplitter as EWS


def data_separators(preparated_data_dict: Dict[str, Union[str, pl.DataFrame]], 
                    parameters: Dict[str, Any]) -> Dict[str, Union[List[int], Type[EWS]]]:
    """Separates the data, using cross-validation, for predictions.

    Arguments:
        prepared_data_dict (Dict[str, Union[str, pl.DataFrame]]): Dictionary with prepared data.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        Dict[str, Union[List[int], Type[EWS]]: Dictionary with data separators.
    """

    transformed_data = preparated_data_dict.get("transformed_data")
    test_size, folds, step_length = parameters.get("TEST_SIZE"), parameters.get("CV_FOLDS"), \
        parameters.get("CV_STEP_LENGTH")

    fh_test = list(range(1, test_size+1))
    initial_window = len(transformed_data) - (test_size + ((folds - 1)*step_length))
    crossval = EWS(fh=fh_test, initial_window=initial_window, step_length=step_length)

    return {
        "fh_test": fh_test,
        "cv": crossval
    }