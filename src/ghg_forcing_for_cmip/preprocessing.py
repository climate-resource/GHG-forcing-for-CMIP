"""
Data preprocessing

Prepare data for statistical analysis to create a GHG forcing dataset.
"""

from enum import Enum
from typing import Union

import numpy as np
import pandas as pd


class Condition(str, Enum):
    """
    Enum for different data conditions

    1. COLLOCATED: both ground-based and satellite data are present
    2. EO_ONLY: only satellite data is present
    3. GB_ONLY: only ground-based data is present
    4. BOTH_NONE: neither ground-based nor satellite data is present
    """

    COLLOCATED = "collocated"
    EO_ONLY = "eo-only"
    GB_ONLY = "gb-only"
    BOTH_NONE = "both-none"


def standardize_feature(series: pd.Series) -> pd.Series:
    """
    Standardize a pandas Series

    Parameters
    ----------
    series :
        pandas Series to be standardized
    """
    return (series - series.mean()) / series.std()


def prepare_dataset(
    df_combined: pd.DataFrame, condition: Union[Condition, str], day: int
) -> pd.DataFrame:
    """
    Prepare datasets for statistical analysis

    Filter data based on the specified condition and
    do required feature engineering for statistical analysis.

    Parameters
    ----------
    df_combined :
        Dataframe combining ground-based and satellite data

    condition :
        condition to filter data:
        "collocated": both ground-based and satellite data are present
        "eo-only": only satellite data is present
        "gb-only": only ground-based data is present
        "both-none": neither ground-based nor satellite data is present

    Returns
    -------
    :
        preprocessed dataframe ready for statistical analysis

    """
    try:
        df_combined.value_gb
    except AttributeError:
        raise AttributeError("DataFrame must contain 'value_gb' column")  # noqa: TRY003

    if condition == "collocated":
        df = df_combined[
            (~df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())
        ]
    elif condition == "eo-only":
        df = df_combined[(df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
    elif condition == "gb-only":
        df = df_combined[(~df_combined.value_gb.isna()) & (df_combined.value_eo.isna())]
    elif condition == "both-none":
        df = df_combined[(df_combined.value_gb.isna()) & (df_combined.value_eo.isna())]
    else:
        raise ValueError(f"{condition} does not exist")  # noqa: TRY003

    df_clean = df.drop_duplicates().reset_index(drop=True)

    df_clean["date"] = pd.to_datetime(df_clean[["year", "month"]].assign(day=day))

    # Feature engineering
    # Seasonal features
    df_clean["season_sin"] = np.sin(2 * np.pi * (df_clean["month"] - 1) / 11)
    df_clean["season_cos"] = np.cos(2 * np.pi * (df_clean["month"] - 1) / 11)

    # Standardize lat, lon, value_eo for easier model convergence
    df_clean["lat_scaled"] = standardize_feature(df_clean["lat"])
    df_clean["lon_scaled"] = standardize_feature(df_clean["lon"])
    df_clean["value_eo_scaled"] = standardize_feature(df_clean["value_eo"])
    df_clean["year_scaled"] = df_clean["year"] - df_clean["year"].min()

    return df_clean


def combine_datasets(
    df_gb: pd.DataFrame,
    df_eo: pd.DataFrame,
    select_cols: list[str] = ["year", "month", "lat", "lon", "value"],
) -> pd.DataFrame:
    """
    Combine ground-based and satellite datasets

    Parameters
    ----------
    df_gb :
        Dataframe containing ground-based data

    df_eo :
        Dataframe containing satellite data

    select_cols :
        Columns to select from both dataframes for merging

    Returns
    -------
    :
        Combined dataframe with ground-based and satellite data
    """
    return (
        df_gb[select_cols]
        .drop_duplicates()
        .merge(
            df_eo[select_cols],
            on=["year", "month", "lat", "lon"],
            how="outer",
            suffixes=("_gb", "_eo"),
        )
    )


def add_hemisphere(df: pd.DataFrame, split_value: Union[float, int]) -> pd.DataFrame:
    """
    Add a grouping variable "hemisphere"

    The levels are defined as follows:
       + southern < - split_value,
       + northern > split_value,
       + -split_value < tropics < split_value

    Parameters
    ----------
    df :
        dataframe including lat variable

    split_value :
        latitudinal value where split in southern,
        northern hemisphere and tropics should be done

    Returns
    -------
    :
        dataframe including new variable hemisphere
    """
    try:
        df.lat
    except AttributeError:
        raise AttributeError("The dataframe has no variable 'lat'.")  # noqa: TRY003

    conditions = [(df["lat"] > split_value), (df["lat"] < -split_value)]
    choices = ["northern", "southern"]
    df["hemisphere"] = np.select(conditions, choices, default="tropics")

    return df


def concat_datasets(dfs: list[pd.DataFrame], obs_gb_value: list[bool]) -> pd.DataFrame:
    """
    Concatenate multiple dataframes

    create additional variable indicating whether data is
    observed or modelled

    Parameters
    ----------
    dfs :
        list of dataframes that shall be concatenated

    obs_gb_value :
        list of booleans indicating for each dataframe in
        `dfs` whether data is observed or not.

    Returns
    -------
    :
        concatenated dataframe with "obs_gb" indicator
        variable
    """
    dfs_prep = []
    for df, obs_gb in zip(dfs, obs_gb_value):
        df["obs_gb"] = obs_gb
        dfs_prep.append(df)

    return pd.concat(dfs_prep)
