"""
combine earth observations and ground-based data in one dataset
"""

import numpy as np
import pandas as pd
import xarray as xr
from prefect import flow, task

from ghg_forcing_for_cmip.data_comparison import CONFIG


@task(
    name="apply_averaging_kernel",
    description="Apply averaging kernel to ground-based data",
    cache_policy=CONFIG.CACHE_POLICIES,
    refresh_cache=True,
)
def apply_averaging_kernel(d_joint_wide: pd.DataFrame, gas: str) -> pd.DataFrame:
    """
    Apply averaging kernel to ground-based data

    according to Buchwitz et al., 2024

    Parameters
    ----------
    d_joint_wide:
        joint dataset in wide format

    gas:
        target greenhouse gas variable

    Returns
    -------
    :
        joint dataset including ground-based value with AK
    """
    # apply averaging kernel to modelled ground-based data
    # (see PUGS, Buchwitz et al. 2024)
    d_joint_wide["ground_based_AK"] = (
        d_joint_wide[f"vmr_profile_{gas}_apriori"]
        + (d_joint_wide["value_vertical"] - d_joint_wide[f"vmr_profile_{gas}_apriori"])
        * d_joint_wide.column_averaging_kernel
    )

    return d_joint_wide


@task(
    name="join_datasets_long",
    description="Join datasets in long format with value-column",
    cache_policy=CONFIG.CACHE_POLICIES,
    refresh_cache=True,
)
def joint_dataset_long(d_joint_AK: pd.DataFrame) -> pd.DataFrame:
    """
    Join satellite and ground-based data in long-format

    Parameters
    ----------
    d_joint_AK:
        joint dataset with averaging kernel applied
        to ground-based data

    Returns
    -------
    :
        joint dataset in long format
    """
    value_columns = [
        "satellite",
        "ground_based",
        "ground_based_no_vertical",
        "ground_based_AK",
    ]
    id_columns = list(set(d_joint_AK.columns).difference(set(value_columns)))
    d_joint_long = pd.melt(
        d_joint_AK, id_vars=id_columns, value_vars=value_columns, var_name="source"
    )

    return d_joint_long


@task(
    name="join_gb-eo_datasets",
    description="Combine earth observations and ground-based data in one dataset",
    cache_policy=CONFIG.CACHE_POLICIES,
    refresh_cache=True,
)
def join_datasets_wide(
    d_vertical: xr.Dataset, path_to_csv: str, gas: str, fillvalue: float = 1e20
) -> pd.DataFrame:
    """
    Joint satellite and ground-based data in wide format

    Parameters
    ----------
    d_vertical:
        dataset with vertical dimension

    path_to_csv:
        path to saved data sets

    gas:
        target greenhouse gas variable

    fillvalue:
        fill values used in OBS4MIPs to indicate NAN

    Returns
    -------
    :
        joint dataset as pandas dataframe
    """
    d_gb = d_vertical
    d_eo = xr.open_dataset(
        path_to_csv
        + f"/{gas}/200301_202212-C3S-L3_X{gas.upper()}-"
        + "GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.5.nc"
    )

    d_eo = d_eo.assign_coords(
        year=d_eo.time.dt.year,
        month=d_eo.time.dt.month,
    )

    d_eo = d_eo.groupby(["year", "month"]).mean()
    d_eo = d_eo.rename({"pressure": "pre"}).assign_coords(pre=d_gb.pre.values[::-1])
    # use NaN instead of fillvalue
    d_eo = d_eo.where(d_eo != fillvalue)
    # convert x-ghg (unitless) to ppb or ppm
    factor = np.where(gas == "co2", 1e6, 1e9)
    for var in [f"x{gas}", f"x{gas}_stddev", f"vmr_profile_{gas}_apriori"]:
        d_eo[var] = d_eo[var] * factor
    ds_merge = xr.combine_by_coords(
        [
            d_eo[
                [
                    f"x{gas}",
                    f"x{gas}_nobs",
                    f"x{gas}_stddev",
                    "column_averaging_kernel",
                    f"vmr_profile_{gas}_apriori",
                ]
            ],
            d_gb,
        ]
    )

    return ds_merge


@flow(
    name="combine_datasets",
)
def join_datasets_flow(path_to_csv: str, gas: str, quantile: float) -> None:
    """
    Run flow to combine satellite and ground-based data

    Parameters
    ----------
    path_to_csv:
        path where datasets are stored

    gas:
        target greenhouse gas variable

    quantile:
        quantile used during binning
    """
    d_vertical = xr.open_dataset(path_to_csv + f"/{gas}/{gas}_vertical_q{quantile}.nc")

    d_joint_wide = join_datasets_wide(d_vertical, path_to_csv=path_to_csv, gas=gas)
    d_joint_AK = apply_averaging_kernel(d_joint_wide, gas=gas)

    d_joint_AK.to_netcdf(path_to_csv + f"/{gas}/{gas}_joint_AK_q{quantile}.nc")
    d_joint_wide.to_netcdf(path_to_csv + f"/{gas}/{gas}_combined_q{quantile}.nc")


if __name__ == "__main__":
    join_datasets_flow("data/downloads", "ch4")
