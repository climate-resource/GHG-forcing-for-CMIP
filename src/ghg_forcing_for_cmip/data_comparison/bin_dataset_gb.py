"""
Compute average concentrations per grid cell
"""

import numpy as np
import pandas as pd
import scipy  # type: ignore
import xarray as xr
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE

CACHE_POLICIES = TASK_SOURCE + INPUTS


@task(
    name="compute_avg_grid_cell",
    description="Compute average ghg-value and std. per grid cell",
    cache_policy=CACHE_POLICIES,
)
def compute_average_per_grid_cell(d_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average ghg-concentration and its std. per grid cell

    Parameters
    ----------
    d_raw:
        raw dataset

    Returns
    -------
    :
        binned pandas dataframe
    """
    d_raw["var"] = d_raw.std_dev**2
    d_raw["pooled_var"] = np.multiply(d_raw["var"], (d_raw.numb - 1.0))

    d_avg = (
        d_raw.groupby(
            ["time", "time_fractional", "year", "month", "lat", "lon", "gas", "unit"]
        )
        .agg({"value": ["mean", "count"], "pooled_var": "sum", "numb": "sum"})
        .reset_index()
    )

    d_avg["pooled_std"] = np.sqrt(
        d_avg[("pooled_var", "sum")]
        / (d_avg[("numb", "sum")] - d_avg[("value", "count")])
    )

    d_avg.drop(
        columns=[("value", "count"), ("pooled_var", "sum"), ("numb", "sum")],
        inplace=True,
    )

    d_avg.columns = pd.Index([col[0] for col in d_avg.columns])
    return d_avg


@task(
    name="select_and_replace_percentile",
    description="Select ghg concentration accord. to percentile",
    cache_policy=CACHE_POLICIES,
)
def select_and_replace_percentile(
    d_binned: pd.DataFrame, quantile: float
) -> pd.DataFrame:
    """
    Select ghg concentration accord. to percentile

    Parameters
    ----------
    d_binned :
        binned pandas dataframe

    quantile :
        quantile used to select the corresponding
        ghg-concentration value

    Returns
    -------
    :
        binned dataframe with ghg-value according to
        quantile
    """
    d_binned = d_binned.copy()
    # in case where pooled_std is NAN (when count=1) use average of
    # the corresponding year as imputation
    std_avg_year = d_binned.groupby("year").agg({"pooled_std": "median"}).reset_index()
    std_avg_year.rename(columns={"pooled_std": "avg_pooled_std"}, inplace=True)
    d_binned = d_binned.merge(std_avg_year, on="year")

    d_binned["pooled_std"] = np.where(
        d_binned.pooled_std.isna(), d_binned.avg_pooled_std, d_binned.pooled_std
    )
    # select respective quantile
    d_binned["value"] = scipy.stats.norm.ppf(
        quantile, loc=d_binned.value, scale=d_binned.pooled_std
    )
    d_binned.drop(columns=["pooled_std", "avg_pooled_std"], inplace=True)

    return d_binned


@flow(name="bin_data_select_quantile", description="Bin dataset and select quantile")
def bin_dataset_flow(path_to_csv: str, gas: str, quantile: float = 0.5) -> None:
    """
    Compute binning of dataset

    Compute average of ghg-concentration (value-mean) and value-std. per grid cell
    Select percentile used as further ghg-concentration which uses
     the rationale: ghg-value = Normal(value-mean, value-std).percentile(p)

    Parameters
    ----------
    path_to_csv :
        path to csv dataset

    gas :
        greenhouse gas target variable

    quantile :
        quantile in percentage (between 0 and 100) to select the
        corresponding ghg-concentration value (used for uncertainty
        quantification)
    """
    d_raw = pd.read_csv(path_to_csv + f"/{gas}/{gas}_raw.csv")

    d_binned = compute_average_per_grid_cell(d_raw=d_raw)

    d_binned_updated = select_and_replace_percentile(
        d_binned=d_binned, quantile=quantile
    )

    ds_binned = xr.Dataset(
        data_vars=dict(
            value=d_binned_updated.set_index(["year", "month", "lat", "lon"])[
                "value"
            ].to_xarray(),
            time=d_binned_updated.set_index(["year", "month", "lat", "lon"])[
                "time"
            ].to_xarray(),
            time_fractional=d_binned_updated.set_index(["year", "month", "lat", "lon"])[
                "time_fractional"
            ].to_xarray(),
        ),
        attrs=dict(
            description="Dataset binned on a 5° by 5° grid.",
            quantile=quantile,
            gas=d_binned_updated.gas.iloc[0],
            unit=d_binned_updated.unit.iloc[0],
        ),
    )

    ds_binned.to_netcdf(path_to_csv + f"/{gas}/{gas}_binned_q{quantile}.nc", mode="w")


if __name__ == "__main__":
    bin_dataset_flow("data/downloads", "ch4", 0.5)
