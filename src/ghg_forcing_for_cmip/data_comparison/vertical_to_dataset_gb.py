"""
model vertical distribution of ground-based data
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from prefect import flow, task

from ghg_forcing_for_cmip.data_comparison import CONFIG
from ghg_forcing_for_cmip.data_comparison.utils import (
    compute_weighted_avg,
)


@task(
    name="add_value_per_pressure",
    description="Add ghg concentration per pressure value",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def add_value_per_pressure(
    d: xr.Dataset, gas: str, pressure_list: npt.NDArray
) -> pd.DataFrame:
    """
    Add ghg concentration per pressure value

    Parameters
    ----------
    d:
        interpolated dataset

    gas:
        target greenhouse gas variable

    pressure_list:
        list of pressure values

    Returns
    -------
    :
        interpolated dataset with ghg-concentration
        per pressure value in wide format
    """
    d2 = d.copy()
    d_res = dict()
    for pressure in pressure_list:
        if gas == "ch4":
            d_res[f"{pressure}"] = np.where(
                pressure > d2.p_tropo,
                d2.value,
                d2.global_1yrs * (pressure / d2.p_tropo) ** d2.scaling_factor,
            )
        if gas == "co2":
            if pressure > 100:  # noqa: PLR2004
                d_res[f"{pressure}"] = d2.value_global_annual + (
                    d2.value - d2.value_global_annual
                ) * ((pressure - 100) / (1000 - 100))
            elif pressure < 100:  # noqa: PLR2004
                d_res[f"{pressure}"] = d2.value_global_5yrs + (
                    d2["100hPa_value"] - d2.value_global_5yrs
                ) * ((pressure - 1) / (100 - 1))
            elif pressure == 100:  # noqa: PLR2004
                d_res[f"{pressure}"] = d2.value_global_annual + (
                    d2.value_global_5yrs - d2.value_global_annual
                ) * (np.sin(d2.lat) ** 2 / 2)
    return d_res


@task(
    name="from_wide_to_long",
    description="Convert dataset from wide to long format",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def from_wide_to_long(
    d: pd.DataFrame,
    id_vars: list[str],
    var_name: str,
    var_values: npt.NDArray = CONFIG.PRESSURE_LIST,
) -> pd.DataFrame:
    """
    Convert dataset from wide to long format

    Parameters
    ----------
    d:
        interpolated dataset in wide format

    id_vars:
        column names that should be maintained
        in long format

    var_name:
        name of the pressure variable

    var_values:
        pressure values list

    Returns
    -------
    :
        dataset in long format
    """
    d1 = d.copy()
    # wide to long format
    d_long = d1.melt(
        id_vars=id_vars,
        var_name=var_name,
        value_name="value_new",
        value_vars=[str(val) for val in var_values],
    )
    d_long["pressure"] = d_long.pressure.astype(np.float32) / 1000.0

    # rename original value
    d_long.rename(
        columns={
            "value": "value_without_vertical",
            "value_new": "value",
            "pressure": "pre",
        },
        inplace=True,
    )

    columns_all = [*id_vars, "value", "value_without_vertical", "pre"]

    return d_long[columns_all]


def add_required_variables(
    d_interpol: xr.Dataset, global_annual_mean: pd.DataFrame, gas: str
) -> pd.DataFrame:
    """
    Add further variables required for computing vertical dimension

    Parameters
    ----------
    d_interpol:
        interpolated data set

    global_annual_mean:
        dataframe with global annual mean and year

    gas:
        target greenhouse gas variable

    Returns
    -------
    :
        interpolated dataset with additional variables
    """
    if gas == "co2":
        # global-annual concentration 5 yrs ago
        d_global_5yrs = global_annual_mean.copy()
        d_global_5yrs["year"] = d_global_5yrs.year + 5

        d_interpol["value_global_5yrs"] = d_global_5yrs.set_index("year")[
            "value"
        ].to_xarray()

        # compute co2 concentration at 100 hPa
        d_interpol["100hPa_value"] = d_interpol.value_global_annual + (
            d_interpol.value_global_5yrs - d_interpol.value_global_annual
        ) * (np.sin(d_interpol.lat) ** 2 / 2)

    if gas == "ch4":
        # global-annual concentration 5 yrs ago
        d_global_1yrs = global_annual_mean.copy()
        d_global_1yrs["year"] = d_global_1yrs.year + 1
        d_interpol["global_1yrs"] = d_interpol["year"].map(
            d_global_1yrs.set_index("year")["value"]
        )
        # compute pressure at tropopause
        d_interpol["p_tropo"] = 250 - 150 * np.cos(d_interpol.lat) ** 2
        # add gas-dependent scaling factor
        d_interpol["scaling_factor"] = np.where(
            abs(d_interpol.lat) >= 45.0,  # noqa: PLR2004
            0.2353 + 0.0225489 * (abs(d_interpol.lat) - 45.0),
            0.2353,
        )

    return d_interpol


@task(
    name="postprocess_vertical_dataset",
    description="clean vertical dataset for further analysis",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def postprocess_vertical_dataset(d_vertical: pd.DataFrame, gas: str) -> pd.DataFrame:
    """
    Postprocess dataset to get final vertical_dataset

    Parameters
    ----------
    d_vertical:
        dataset after modelling the vertical dimension

    gas:
        target greenhouse gas variable

    Returns
    -------
    :
        clean vertical dataset
    """
    if gas == "co2":
        d_vertical.drop(
            columns=["global_annual", "global_5yrs", "100hPa_value", "Unnamed: 0"],
            inplace=True,
        )
    elif gas == "ch4":
        d_vertical.drop(
            columns=[
                "global_annual",
                "global_1yrs",
                "p_tropo",
                "scaling_factor",
                "Unnamed: 0",
            ],
            inplace=True,
        )

    return d_vertical


@flow(name="add_vertical")
def add_vertical_flow(path_to_csv: str, gas: str) -> None:
    """
    Add vertical distribution to ghg concentration data

    Parameters
    ----------
    path_to_csv : str
        path to interpolated dataset

    gas : str
        target greenhouse gas variable
    """
    # d_interpol = pd.read_csv(path_to_csv + f"/{gas}/{gas}_interpolated.csv")
    d_interpol = xr.open_dataset(path_to_csv + f"/{gas}/{gas}_interpolated.nc")

    ds_temp = d_interpol.copy()
    df_temp = (
        ds_temp.to_dataframe()
        .reset_index()
        .drop(columns="value")
        .rename(columns={"value_interpolated": "value"})
    )

    # compute global-annual surface concentration
    global_annual_mean = compute_weighted_avg(df_temp, ["year"])

    ds_temp["value_global_annual"] = global_annual_mean.set_index(["year"])[
        "value"
    ].to_xarray()

    # compute further required variables for computing
    # vertical dimension
    add_required_variables(ds_temp, global_annual_mean, gas)

    # compute concentration per pressure level
    d_pressure = add_value_per_pressure(ds_temp, gas, CONFIG.PRESSURE_LIST)

    vars_to_stack = list(d_pressure.keys())
    da_merged = xr.concat([d_pressure[var] for var in vars_to_stack], dim="pre")
    da_merged = da_merged.assign_coords(pre=vars_to_stack)

    d_interpol["value_vertical"] = da_merged
    d_interpol.to_netcdf(path_to_csv + f"/{gas}/{gas}_vertical.nc")


if __name__ == "__main__":
    add_vertical_flow("data/downloads", "co2")
