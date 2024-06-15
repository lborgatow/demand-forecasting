from typing import List, Dict, Any

import polars as pl


def format_digit(df: pl.DataFrame, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Format the digits of the variable records that will form the unique_id.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data.
        parameters (Dict[str, Any]): Dictionary with global parameters. 

    Returns:
        pl.DataFrame: DataFrame with formatted variables.
    """
    
    database = parameters.get("DATABASE")
    
    if database == "STORE_ITEM":
        return df.with_columns(
            store=pl.col("store").cast(str).str.zfill(2),
            item=pl.col("item").cast(str).str.zfill(2)
        )
    elif database == "FOOD":
        return df.with_columns(
            center=pl.col("center_id").cast(str).str.zfill(3),
            meal=pl.col("meal_id").cast(str).str.zfill(4)
        ).drop(["center_id", "meal_id"])


def create_unique_id(df: pl.DataFrame, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Create a unique identifier for the items to forecast. 

    Args:
        df (pl.DataFrame): DataFrame with the temporal data.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: Dataframe with the unique identifier included.
    """

    database = parameters.get("DATABASE")
    
    if database == "STORE_ITEM":
        return df.with_columns(
            pl.concat_str(["store", "item"], separator="_").alias("unique_id")
        )
        
    elif database == "FOOD":
        return df.with_columns(
        pl.concat_str(["center", "meal"], separator="_").alias("unique_id")
    )


def filter_uids_by_amount(df: pl.DataFrame, min_amount: int) -> pl.DataFrame:
    """Filter unique identifiers by amount of data. Products must have at least 
    "min_amount" of data for prediction to be realized.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data and the unique 
        identifier.
        min_amount (int): Minimum amount of data required.

    Returns:
        pl.DataFrame: DataFrame filtered by the amount of data.
    """
    
    df = df.with_columns(count=pl.count("unique_id").over("unique_id"))
    df = df.filter(pl.col("count") >= min_amount)

    return df.drop("count")


def filter_uids_by_week(df: pl.DataFrame) -> pl.DataFrame:
    """Filter unique ids by the first and the last orders records.
    Unique ids that have not been ordered in the first week and for more than 
    2 months (~8 weeks) will be filtered and eliminated from the forecasting process.

    Args:
        df (pl.DataFrame): DataFrame with temporal data and identifier single.

    Returns:
        pl.DataFrame: DataFrame filtered by last sale date.
    """

    min_week = df["week"].min()
    max_week = df["week"].max()
    limit_weeks = 8

    unique_id_filtered = (
        df.group_by("unique_id")
        .agg([
            pl.col("week").min().alias("startDate"),
            pl.col("week").max().alias("endDate")
        ])
        .filter(
            (pl.col("startDate") == min_week) & 
            (pl.col("endDate") > max_week - limit_weeks)
        )
        .select("unique_id")
    )

    return df.filter(pl.col("unique_id").is_in(unique_id_filtered["unique_id"].unique()))


def create_df_with_all_weeks(df: pl.DataFrame) -> pl.DataFrame:
    """Insert zero date of sale for unique ids that were not sold in the week. 
    If any unique ID does not get orders in the week, a null value will be inserted on that date.

    Args:
        df (pl.DataFrame): DataFrame with temporal data and unique identifier.

    Returns:
        pl.DataFrame: DataFrame with the addition of null sales dates.
    """

    min_week = df["week"].min()
    max_week = df["week"].max()

    weeks_df = pl.DataFrame({"week": list(range(min_week, max_week + 1))})
    
    return (
        df.select("unique_id")
        .unique(maintain_order=True)
        .join(weeks_df, how="cross")
        .join(df, how="left", on=["unique_id", "week"])
    )


def fill_null_values(df: pl.DataFrame, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Replace Null Values ​​inserted in the function
    "create_df_with_all_weeks" by Zero Values.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data and the
        unique identifier.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with zero sales replaced by zero.
    """

    database = parameters.get("DATABASE")
    
    if database == "FOOD":
        df = df.with_columns(
            pl.col("num_orders").fill_null(0),
        )
        
        df[["center", "meal"]] = (
            df["unique_id"]
            .str.split_exact("_", 1)
            .to_frame()
            .unnest("unique_id")
        )

    return df


def create_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Create fictitious dates based on weeks for model training.

    Args:
        df (pl.DataFrame): DataFrame with the temporal data
        with the weeks.

    Returns:
        pl.DataFrame: DataFrame with the fictitious dates.
    """
    
    initial_date = pl.datetime(year=2021, month=1, day=3)
    dates = initial_date + (df["week"] - 1) * pl.duration(weeks=1)

    return df.with_columns(
        fictitious_dates=dates.cast(pl.Datetime)
    )


def format_columns(df: pl.DataFrame, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Format columns.

    Args:
        df (pl.DataFrame): DataFrame with temporal data.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame with columns formatted.
     """
    
    database = parameters.get("DATABASE")
    
    if database == "STORE_ITEM":
        return df.with_columns(
            pl.col("date").str.to_datetime().alias("ds"),
            pl.col("sales").alias("y")
        ).drop(["date", "sales"])
        
    elif database == "FOOD":
        return df.with_columns(
            pl.col("fictitious_dates").alias("ds"),
            pl.col("num_orders").alias("y")
        ).drop(["fictitious_dates", "num_orders"])


def aggregate_days_into_weeks(df: pl.DataFrame, add_days: int) -> pl.DataFrame:
    """Aggregate daily demands into weekly demands. 

    Args:
        df (pl.DataFrame): DataFrame with daily data.
        add_days (int): Days needed to add to the dates so that they are 
        compatible with the pandas weekly frequency ("W").

    Returns:
        pl.DataFrame: DataFrame aggregated into weekly data.
    """
    
    agg_df = (
        df.group_by_dynamic(
            index_column="ds", 
            every="1w",
            closed="left", 
            group_by="unique_id",
            include_boundaries=False
        )
        .agg(
            [
                pl.first("store"),
                pl.first("item"),
                pl.sum("y")
            ]
        )
    )

    return agg_df.with_columns(
        (pl.col("ds") + pl.duration(days=add_days)).alias("ds")
    )


def process_data(data: pl.DataFrame, parameters: Dict[str, Any]) -> pl.DataFrame:
    """Process the data before starting predictions.

    Args:
        data (pl.DataFrame): DataFrame with temporal data.
        parameters (Dict[str, Any]): Dictionary with global parameters.

    Returns:
        pl.DataFrame: DataFrame processed.
    """
    
    database = parameters.get("DATABASE")
    frequency = parameters.get("FREQUENCY")
    
    if database == "STORE_ITEM":
        df = format_digit(data, parameters)
        df = create_unique_id(df, parameters)
        df = format_columns(df, parameters)
        df = df.with_columns(
            pl.col("y").cast(pl.Int32).alias("y")
        )
        if frequency == "WEEKLY": 
            df = aggregate_days_into_weeks(df, 6)
        return df.select(["ds", "unique_id", "store", "item", "y"])       
        
        
    elif database == "FOOD":
        df = format_digit(data, parameters)
        df = create_unique_id(df, parameters)
        df = filter_uids_by_amount(df, 104)
        df = filter_uids_by_week(df)
        df = create_df_with_all_weeks(df)
        df = fill_null_values(df, parameters)
        df = create_dates(df)
        df = format_columns(df, parameters)
        df = df.with_columns(
            pl.col("week").cast(pl.Int32),
            pl.col("y").cast(pl.Int32)
        )
        return df.select(["week", "ds", "unique_id", "center", "meal", "y"])
        


def get_unique_ids(data: pl.DataFrame) -> List[str]:
    """Get the list containing all unique ids for the predictions.

    Args:
        data (pl.DataFrame): DataFrame with the data.

    Returns:
        List[str]: List of unique_ids.
    """

    return data["unique_id"].unique().sort().to_list()

