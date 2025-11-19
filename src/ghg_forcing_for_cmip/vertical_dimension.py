"""
model vertical dimension of GHG concentrations
"""

from enum import Enum

import pandas as pd
from prefect import flow, task

from ghg_forcing_for_cmip import CONFIG


class Method(Enum):
    """
    Methods used for constructing vertical dimension
    """

    MEINSHAUSEN2017 = 1


@task(
    name="method_meinshausen2017",
    description="use method described in Meinshausen et al. 2017",
    task_run_name="add_vertical_meinshausen2017",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def method_meinshausen2017(data: pd.DataFrame) -> None:
    """
    Construct vertical according to Meinshausen et al., 2017

    References
    ----------
    Meinshausen, et al. (2017). Historical greenhouse
    gas concentrations for climate modelling (CMIP6),
    Geosci. Model Dev., 10, 2057-2116,
    https://doi.org/10.5194/gmd-10-2057-2017.

    Parameters
    ----------
    data :
        binned data set
    """
    surface_pressure_hPa = 1000  # noqa: F841


@flow(name="add_vertical")
def add_vertical(method: Method):
    """
    Construct vertical dimension according to method

    Parameters
    ----------
    method :
        method used to construct vertical
        dimension for ground-based data
    """
    if method is Method.MEINSHAUSEN2017:
        method_meinshausen2017()
