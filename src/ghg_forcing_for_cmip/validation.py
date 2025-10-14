"""
In this module the validation schemes are stored

used to validate columns in the created dataframe
after scraping of data
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


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
    year: Series[int] = pa.Field(ge=1968, le=datetime.now().year)
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


def compute_discrepancy_collocated(d: pd.DataFrame, gas: str, measure: str) -> Any:
    """
    Compute rmse per site code in collocated data

    Parameters
    ----------
    d :
        data frame with collocated data

    gas :
        either "ch4" or "co2"

    measure:
        either "rmse" or "dcor";
        rmse: stands for root mean squared error
        d_cor: refers to the correlation difference (1-cor)

    Returns
    -------
    :
        dataframe with rmse information
    """
    if measure not in ["rmse", "dcor"]:
        raise ValueError(  # noqa: TRY003
            f"measure must be either 'rmse' or 'dcor', but got {measure}"
        )

    if measure == "rmse":
        res = (
            d.groupby("site_code")[["site_code", "value_eo", "value_gb"]]
            .apply(lambda d: np.sqrt(np.mean((d.value_eo - d.value_gb) ** 2)))
            .reset_index()
            .rename(columns={0: "rmse_" + gas})
        )

    if measure == "dcor":
        res = (
            d.groupby("site_code")[["site_code", "value_eo", "value_gb"]]
            .apply(lambda d: 1 - abs(np.corrcoef(d.value_eo, d.value_gb)[0, 1]))
            .reset_index()
            .rename(columns={0: "dcor_" + gas})
        )

    return res
