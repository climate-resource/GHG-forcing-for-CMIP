"""
Global helper functions
"""

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series
from prefect.tasks import task_input_hash


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


class GroundDataSchema(pa.DataFrameModel):
    """
    validate columns of dataset combining data from NOAA and (A)GAGE

    """

    time_fractional: Series[np.float64]
    time: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"}
    )
    year: Series[np.int64] = pa.Field(ge=1968, le=datetime.now().year)
    month: Series[np.int64] = pa.Field(ge=1, le=12)
    latitude: Series[np.float64] = pa.Field(ge=-90, le=90)
    longitude: Series[np.float64] = pa.Field(ge=-180, le=180)
    lat_bnd: Series[np.int64] = pa.Field(ge=-90, le=90)
    lon_bnd: Series[np.int64] = pa.Field(ge=-180, le=180)
    lat: Series[np.float64] = pa.Field(gt=-90, lt=90)
    lon: Series[np.float64] = pa.Field(gt=-180, lt=180)
    value: Series[np.float64] = pa.Field(gt=0.0, lt=9999.0)
    std_dev: Series[np.float64] = pa.Field(ge=0)
    numb: Series[np.int64] = pa.Field(gt=0)
    site_code: Series[str]
    network: Series[str] = pa.Field(isin=["agage", "gage", "noaa"])
    altitude: Series[np.float64] = pa.Field(ge=0.0, nullable=True)
    insitu_vs_flask: Series[str] = pa.Field(isin=["flask", "insitu"], nullable=True)
    sampling_strategy: Series[str] = pa.Field(
        isin=["surface-flask", "shipboard-flask", "surface-insitu"], nullable=True
    )
    gas: Series[str] = pa.Field(isin=["ch4", "co2"])
    unit: Series[str] = pa.Field(isin=["ppb", "ppm"])
