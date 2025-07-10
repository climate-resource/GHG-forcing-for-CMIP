"""
Compute average concentrations per grid cell
"""

import numpy as np
import pandas as pd
import scipy  # type: ignore
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE

CACHE_POLICIES = TASK_SOURCE + INPUTS


@task(
    name="compute_avg_grid_cell",
    description="Compute average ghg-value and std. per grid cell",
    cache_policy=CACHE_POLICIES,
)
def compute_average_per_grid_cell(path_to_csv: str, gas: str) -> pd.DataFrame:
    """
    Compute average ghg-concentration and its std. per grid cell

    Parameters
    ----------
    path_to_csv:
        path to saved dataset

    gas :
        greenhouse gas target variable

    Returns
    -------
    :
        binned pandas dataframe
    """
    d = pd.read_csv(path_to_csv + f"/{gas}/{gas}_raw.csv")
    d["var"] = d.std_dev**2
    d["pooled_var"] = np.multiply(d["var"], (d.numb - 1.0))

    d_avg = (
        d.groupby(
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
        quantile in percentage (between 0 and 1) to select the
        corresponding ghg-concentration value (used for uncertainty
        quantification)
    """
    d_binned = compute_average_per_grid_cell(path_to_csv=path_to_csv, gas=gas)

    d_binned_updated = select_and_replace_percentile(
        d_binned=d_binned, quantile=quantile
    )

    d_binned_updated.to_csv(path_to_csv + f"/{gas}/{gas}_binned.csv", index=False)


if __name__ == "__main__":
    bin_dataset_flow("data/downloads", "ch4", 0.5)
