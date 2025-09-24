"""
In this module the validation schemes are stored

used to validate columns in the created dataframe
after scraping of data
"""

from datetime import datetime

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
