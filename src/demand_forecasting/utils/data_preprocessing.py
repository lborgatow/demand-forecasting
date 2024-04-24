from typing import List

import polars as pl


def format_single_digit(df: pl.DataFrame) -> pl.DataFrame:
    """Format the records of the variables "store" and "item" 
    that have only one digit.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data 
        of the store and items.

    Returns:
        pl.DataFrame: DataFrame with formatted variables.
    """
    
    return df.with_columns(
        store=pl.col("store").cast(str).str.zfill(2),
        item=pl.col("item").cast(str).str.zfill(2)
    )


def create_unique_id(df: pl.DataFrame) -> pl.DataFrame:
    """Create a unique identifier for the items to forecast. 
    This function combines two codes: the store code and the item code.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data of the stores 
        and items.

    Returns:
        pl.DataFrame: Dataframe with the unique identifier included.
    """

    return df.with_columns(
        pl.concat_str(["store", "item"], separator="_").alias("unique_id")
    )


def format_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Format column names.

    Args:
        df (pl.DataFrame): DataFrame with temporal data.

    Returns:
        pl.DataFrame: DataFrame with column names formatted.
     """
    
    return df.with_columns(
        pl.col("date").str.to_datetime().alias("ds"),
        pl.col("sales").alias("y")
    ).drop(["date", "sales"])


def process_data(data: pl.DataFrame) -> pl.DataFrame:
    """Process the data before starting predictions.

    Args:
        data (pl.DataFrame): DataFrame with temporal data.

    Returns:
        pl.DataFrame: DataFrame processed.
    """

    df = format_single_digit(data)
    df = create_unique_id(df)
    df = format_column_names(df)

    return df.select(["ds", "unique_id", "store", "item", "y"])


def get_unique_ids(data: pl.DataFrame) -> List[str]:
    """Get the list containing all unique ids for the predictions.

    Args:
        data (pl.DataFrame): DataFrame with the data.

    Returns:
        List[str]: List of unique_ids.
    """

    return data["unique_id"].unique().sort().to_list()

