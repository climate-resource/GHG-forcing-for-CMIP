"""
Global helper functions
"""

import io
import os
import zipfile
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
import requests
from pandera.typing.pandas import Series
from prefect import task
from prefect.tasks import task_input_hash

from . import CONFIG


def ensure_trailing_slash(path: str) -> str:
    """
    Ensure trailing slash at the end of a path

    Parameters
    ----------
    path
        the path / directory

    Returns
    -------
    :
        path with trailing slash
    """
    return path if path.endswith("/") else path + "/"


def is_pytest_running() -> bool:
    """
    Check whether Pytest is running
    """
    return "PYTEST_CURRENT_TEST" in os.environ


def custom_cache_key_fn() -> Optional[Any]:
    """
    Check whether results should be cached

    If Pytest is running, don't cache results
    otherwise use task_input_hash
    """
    if is_pytest_running():
        # Disable caching during pytest by returning None
        return None
    else:
        # Normal caching key, e.g. hash of inputs
        return task_input_hash


def compute_weighted_avg(
    d: pd.DataFrame, grouping_vars: list[str], value_name: str = "value"
) -> pd.DataFrame:
    """
    Compute the weighted average of ghg concentrations

    weighted according to area

    Parameters
    ----------
    d:
        dataset with ghg concentrations and lat, lon
        information

    grouping_vars:
        list of variables to group by or that should be
        maintained

    value_name:
        name of the value variable

    Returns
    -------
    :
        dataset with weighted ghg concentrations
    """
    d1 = d.copy()

    # Define constants
    delta_deg = CONFIG.GRID_CELL_SIZE
    deg2rad = np.deg2rad

    # Calculate bounds
    lat_bnd_0 = deg2rad(d1.lat + delta_deg)
    lat_bnd_1 = deg2rad(d1.lat - delta_deg)
    lon_bnd_0 = deg2rad(d1.lon + delta_deg)
    lon_bnd_1 = deg2rad(d1.lon - delta_deg)

    # weighted average
    delta_lat = np.abs(np.sin(lat_bnd_0) - np.sin(lat_bnd_1))
    delta_lon = np.abs(lon_bnd_0 - lon_bnd_1)
    d1["weight"] = delta_lon * delta_lat

    # Compute weighted value
    d1["value_weighted"] = d1[value_name] * d1.weight
    d1.dropna(subset="value_weighted", inplace=True)

    d2 = (
        d1.groupby(grouping_vars)
        .agg({"weight": "sum", "value_weighted": "sum"})
        .reset_index()
    )

    d2[value_name] = d2.value_weighted / d2.weight
    d2.drop(columns=["value_weighted", "weight"], inplace=True)
    return d2


class GroundDataSchema(pa.DataFrameModel):
    """
    validate columns of dataset combining data from NOAA and (A)GAGE

    """

    time_fractional: Series[float]
    time: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"}
    )
    year: Series[int] = pa.Field(ge=1960, le=datetime.now().year)
    month: Series[int] = pa.Field(ge=1, le=12)
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    longitude: Series[float] = pa.Field(ge=-180, le=180)
    lat_bnd: Series[int] = pa.Field(ge=-90, le=90)
    lon_bnd: Series[int] = pa.Field(ge=-180, le=180)
    lat: Series[float] = pa.Field(gt=-90, lt=90)
    lon: Series[float] = pa.Field(gt=-180, lt=180)
    value: Series[float] = pa.Field(gt=0.0, lt=9999.0, nullable=True)
    std_dev: Series[float] = pa.Field(ge=0, nullable=True)
    numb: Series[int] = pa.Field(ge=0, nullable=True)
    site_code: Series[str]
    network: Series[str] = pa.Field(isin=["agage", "gage", "noaa"])
    altitude: Series[float] = pa.Field(ge=0.0, nullable=True)
    insitu_vs_flask: Series[str] = pa.Field(isin=["flask", "insitu"], nullable=True)
    sampling_strategy: Series[str] = pa.Field(nullable=True)
    gas: Series[str] = pa.Field(isin=["ch4", "co2"])
    unit: Series[str] = pa.Field(isin=["ppb", "ppm"])
    version: Series[str] = pa.Field(nullable=True)
    instrument: Series[str] = pa.Field(nullable=True)


class EODataSchema(pa.DataFrameModel):
    """
    validate columns of satellite dataset

    """

    time: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"}
    )
    year: Series[int] = pa.Field(ge=1960, le=datetime.now().year)
    month: Series[int] = pa.Field(ge=1, le=12)
    lat_bnd: Series[int] = pa.Field(ge=-90, le=90)
    lon_bnd: Series[int] = pa.Field(ge=-180, le=180)
    bnd: Series[int] = pa.Field(isin=[0, 1])
    lat: Series[float] = pa.Field(gt=-90, lt=90)
    lon: Series[float] = pa.Field(gt=-180, lt=180)
    value: Series[float] = pa.Field(gt=0.0, lt=9999.0)
    std_dev: Series[float] = pa.Field(ge=0)
    numb: Series[int] = pa.Field(gt=0)
    gas: Series[str] = pa.Field(isin=["ch4", "co2"])
    unit: Series[str] = pa.Field(isin=["ppb", "ppm"])
    pre: Series[float] = pa.Field(gt=0, lt=1)
    column_averaging_kernel: Series[float]
    vmr_profile_apriori: Series[float]


@task(name="save_dataset")
def save_data(d: pd.DataFrame, path_to_save: str, gas: str, dataset_name: str) -> None:
    """
    Save dataset to disk

    Parameters
    ----------
    d:
        dataset to save

    path_to_save:
        path to save dataset

    gas:
        target greenhouse gas variable

    dataset_name:
        suffix for file name to discriminate dataset files
    """
    d.to_csv(path_to_save + f"/{gas}/{gas}_{dataset_name}.csv")


def download_data() -> None:
    """
    Download data for jupyter notebooks in documentation

    Returns
    -------
    :
        downloaded data (ch4, co2)
    """
    link = "https://github.com/climate-resource/ghg-forcing-for-cmip/releases/download/"
    release_version = "v0.1.0-alpha"
    url_ch4 = link + release_version + "/ch4.zip"
    url_co2 = link + release_version + "/co2.zip"

    if not os.path.exists("docs/data"):
        os.makedirs("docs/data", exist_ok=True)
        for url in [url_ch4, url_co2]:
            response = requests.get(url)  # noqa: S113

            # Open the zip file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
                # Loop through each file inside the zip
                for filename in thezip.namelist():
                    if filename.endswith(".csv"):
                        with thezip.open(filename) as file:
                            # Read CSV into pandas dataframe
                            pd.read_csv(file).to_csv(
                                f"docs/data/{filename}", index=False
                            )


def load_data(dataset_name: str) -> pd.DataFrame:
    """
    Load datasets from release assets for documentation

    Parameters
    ----------
    dataset_name:
        dataset to load (incl. file-ending .csv)

    Returns
    -------
    :
        loaded pandas dataframe
    """
    download_data()

    return pd.read_csv("docs/data/" + dataset_name)
