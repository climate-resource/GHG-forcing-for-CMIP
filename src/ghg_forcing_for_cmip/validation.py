"""
In this module the validation schemes are stored

used to validate columns in the created dataframe
after scraping of data
"""

from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


class GroundDataRow(BaseModel):
    """
    Validates a single row of ground station data (NOAA/AGAGE).
    """

    # Time handling: Pydantic will parse standard datetime strings or objects
    time: datetime
    time_fractional: float

    # Dynamic validator for year is below
    year: int = Field(ge=1960)
    month: int = Field(ge=1, le=12)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    lat_bnd: int = Field(ge=-90, le=90)
    lon_bnd: int = Field(ge=-180, le=180)

    # Strictly greater/less than (gt/lt)
    lat: float = Field(gt=-90, lt=90)
    lon: float = Field(gt=-180, lt=180)

    # Nullable fields (Optional)
    value: Optional[float] = Field(default=None, gt=0.0, lt=9999.0)
    std_dev: Optional[float] = Field(default=None, ge=0.0)
    numb: Optional[int] = Field(default=None, ge=0)

    site_code: str
    network: Literal["agage", "gage", "noaa"]

    altitude: Optional[float] = Field(default=None, ge=0.0)
    insitu_vs_flask: Optional[Literal["flask", "insitu"]] = None
    sampling_strategy: Optional[str] = None

    gas: Literal["ch4", "co2"]
    unit: Literal["ppb", "ppm"]

    version: Optional[str] = None
    instrument: Optional[str] = None

    # Validator to ensure year is not in the future
    @field_validator("year")
    @classmethod
    def validate_year_le_now(cls, v: int) -> int:
        """Validate year is not in the future"""
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f"Year must be <= {current_year}, got {v}")  # noqa: TRY003
        return v


class EODataRow(BaseModel):
    """
    Validates a single row of satellite data.
    """

    # Use Field constraints for numeric ranges
    year: int = Field(ge=1968)
    month: int = Field(ge=1, le=12)

    lat_bnd: int = Field(ge=-90, le=90)
    lon_bnd: int = Field(ge=-180, le=180)

    # Use Literal for strict equality checks (equivalent to pandera `isin`)
    bnd: Literal[0, 1]

    lat: float = Field(gt=-90, lt=90)
    lon: float = Field(gt=-180, lt=180)

    value: float = Field(gt=0.0, lt=9999.0)
    std_dev: float = Field(ge=0.0)
    numb: int = Field(gt=0)

    gas: Literal["ch4", "co2"]
    unit: Literal["ppb", "ppm"]

    pre: float = Field(gt=0.0, lt=1.0)
    column_averaging_kernel: float
    vmr_profile_apriori: float

    # Custom validator for dynamic checks (like current year)
    @field_validator("year")
    @classmethod
    def validate_year_le_now(cls, v: int) -> int:
        """Validate year is not in the future"""
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f"Year must be <= {current_year}, got {v}")  # noqa: TRY003
        return v


def validate_gb_dataframe(df: pd.DataFrame) -> list[GroundDataRow]:
    """
    Iterate over DataFrame rows and validates them against the Pydantic model.
    """
    # Convert DataFrame to a list of dictionaries
    records = df.to_dict(orient="records")

    try:
        # List comprehension triggers validation for every row
        for record in records:
            GroundDataRow(**record)

    except ValidationError as e:
        print("GB dataframe validation failed.")
        print(e)
        raise


def validate_eo_dataframe(df: pd.DataFrame) -> list[EODataRow]:
    """
    Iterate over DataFrame rows and validates them against the Pydantic model.
    """
    # Convert DataFrame to a list of dictionaries
    records = df.to_dict(orient="records")

    try:
        # List comprehension triggers validation for every row
        for record in records:
            EODataRow(**record)

    except ValidationError as e:
        print("EO dataframe validation failed.")
        print(e)
        raise


def compute_discrepancy_collocated(d: pd.DataFrame, gas: str, measure: str):
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
        dataframe with measure information
    """
    if measure not in ["rmse", "dcor"]:
        raise ValueError(  # noqa: TRY003
            f"measure must be either 'rmse' or 'dcor', but got {measure}"
        )

    if measure == "rmse":
        res = (
            d.groupby("site_code")[["site_code", "value_eo", "value_gb"]]
            .apply(
                lambda g: pd.Series(
                    {
                        "rmse_" + gas: np.sqrt(np.mean((g.value_eo - g.value_gb) ** 2)),  # type: ignore
                        "bias_" + gas: np.mean(g.value_eo - g.value_gb),  # type: ignore
                        "var_" + gas: np.var(g.value_eo - g.value_gb),  # type: ignore
                    }
                )
            )
            .reset_index()
            .rename(columns={0: "rmse_" + gas})
        )

    if measure == "dcor":
        res = (
            d.groupby("site_code")[["site_code", "value_eo", "value_gb"]]
            .apply(lambda d: 1 - abs(np.corrcoef(d.value_eo, d.value_gb)[0, 1]))  # type: ignore
            .reset_index()
            .rename(columns={0: "dcor_" + gas})
        )

    return res
