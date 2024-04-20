from typing import List

import polars as pl


def create_unique_id(df: pl.DataFrame) -> pl.DataFrame:
    """Create a unique identifier for the items to forecast. 
    This function combines two codes: the store code and the item code.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data of the items.

    Returns:
        pl.DataFrame: Dataframe with the unique identifier included.
    """

    return df.with_columns(
        pl.concat_str(["store", "item"], separator="_").alias("unique_id")
    )


def process_data(data: pl.DataFrame) -> pl.DataFrame:
    """Process the data before starting predictions.

    Args:
        data (pl.DataFrame): DataFrame with temporal data.

    Returns:
        pl.DataFrame: DataFrame processed.
    """

    df = create_unique_id(data)
    df = df.with_columns(df["date"].str.to_datetime())

    return df.select(["date", "unique_id", "store", "item", "sales"])


def get_unique_ids(data: pl.DataFrame) -> List[str]:
    """Get the list containing all unique ids for the predictions.

    Args:
        data (pl.DataFrame): DataFrame with the data.

    Returns:
        List[str]: List of unique_ids.
    """

    return data["unique_id"].unique().sort().to_list()

