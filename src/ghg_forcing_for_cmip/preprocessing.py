"""
Data preprocessing

Prepare data for statistical analysis to create a GHG forcing dataset.
"""

import itertools
from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt
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
    select_cols: list[str] = ["year", "month", "lat", "lon"],
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
    df_gb_agg = df_gb.groupby(select_cols).agg({"value": "mean"}).reset_index()
    df_eo_agg = df_eo.groupby(select_cols).agg({"value": "mean"}).reset_index()

    return df_gb_agg.merge(
        df_eo_agg,
        on=select_cols,
        how="outer",
        suffixes=("_gb", "_eo"),
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


def add_missing_lat_lon_combinations(  # noqa: PLR0913
    df: pd.DataFrame,
    year_seq: npt.NDArray,
    grid_cell_size: int,
    day: int,
    months: int = 12,
    max_lat: int = 90,
    max_lon: int = 180,
    selected_vars: list[str] = ["year", "month", "lat", "lon"],
) -> pd.DataFrame:
    """
    Add all missing latitude-longitude combinations to the data grid

    1. create template data grid with all possible combinations
    2. merge template data grid with observed data

    Parameters
    ----------
    df :
        observed data

    year_seq :
        array with years

    grid_cell_size :
        size of grid cell

    day :
        day used for creating a date

    months :
        number of months in a year,
        by default 12

    max_lat :
        maximum absolute latitudinale value,
        by default 90

    max_lon :
        maximum absolute longitudinale value,
        by default 180

    selected_vars :
        relevant data variables for merging,
        by default ["year", "month", "lat", "lon"]

    Returns
    -------
    :
        data set with all possible lat x lon combination
    """
    df_template = pd.DataFrame(
        itertools.product(
            year_seq,
            np.arange(1, int(months + 1), 1),
            np.arange(-(max_lat - (grid_cell_size / 2)), max_lat, grid_cell_size),
            np.arange(-(max_lon - (grid_cell_size / 2)), max_lon, grid_cell_size),
        ),
        columns=selected_vars,
    )

    # combine dataset with template data to get also all combinations
    # where both, GB and EO are NaN
    df_total = df.merge(df_template, on=selected_vars, how="outer")
    df_total["date"] = pd.to_datetime(df_total[["year", "month"]].assign(day=day))

    return df_total.set_index("date")


def preprocess_prediction_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset for prediction task

    feature engineering and scaling

    Parameters
    ----------
    df :
        raw dataset that shall be used
        for fitting prediction model

    Returns
    -------
    :
        dataset prepared for prediction task
    """
    df_feat = df.copy()

    month_decimal = (df_feat["month"] - 1) / 12

    # cyclical time features
    df_feat["month_sin"] = np.sin(2 * np.pi * month_decimal)
    df_feat["month_cos"] = np.cos(2 * np.pi * month_decimal)

    # spatial coordinates
    # converts Lat/Lon to x, y, z to represent the spherical globe accurately
    # convert degrees to radians first
    lat_rad = np.radians(df_feat["lat"])
    lon_rad = np.radians(df_feat["lon"])

    df_feat["x_coord"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_feat["y_coord"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_feat["z_coord"] = np.sin(lat_rad)

    df_feat["decimal_year"] = df_feat["year"] + month_decimal
    df_feat["lat_x_year"] = df_feat["decimal_year"] * df_feat["lat"]

    # captures non-linear growth (improves slope accuracy for future)
    df_feat["year_squared"] = df_feat["decimal_year"] ** 2

    return df_feat
