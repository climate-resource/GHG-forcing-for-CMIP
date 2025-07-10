"""
combine earth observations and ground-based data in one dataset
"""

import pandas as pd
from prefect import flow, task

from ghg_forcing_for_cmip.data_comparison import CONFIG
from ghg_forcing_for_cmip.data_comparison.utils import save_data


@task(
    name="apply_averaging_kernel",
    description="Apply averaging kernel to ground-based data",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def apply_averaging_kernel(d_joint_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply averaging kernel to ground-based data

    according to Buchwitz et al., 2024

    Parameters
    ----------
    d_joint_wide:
        joint dataset in wide format

    Returns
    -------
    :
        joint dataset including ground-based value with AK
    """
    # apply averaging kernel to modelled ground-based data
    # (see PUGS, Buchwitz et al. 2024)
    d_joint_wide["ground_based_AK"] = (
        d_joint_wide["vmr_profile_apriori"]
        + (d_joint_wide["ground_based"] - d_joint_wide["vmr_profile_apriori"])
        * d_joint_wide.column_averaging_kernel
    )

    return d_joint_wide


@task(
    name="join_datasets_long",
    description="Join datasets in long format with value-column",
    cache_policy=CONFIG.CACHE_POLICIES,
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
)
def join_datasets_wide(path_to_csv: str, gas: str) -> pd.DataFrame:
    """
    Joint satellite and ground-based data in wide format

    Parameters
    ----------
    path_to_csv:
        path to saved data sets

    gas:
        target greenhouse gas variable

    Returns
    -------
    :
        joint dataset as pandas dataframe
    """
    d_gb = pd.read_csv(path_to_csv + f"/{gas}/{gas}_vertical.csv")
    d_eo = pd.read_csv(path_to_csv + f"/{gas}/{gas}_eo_raw.csv")

    d_gb.rename(
        columns={
            "value": "ground_based",
            "value_without_vertical": "ground_based_no_vertical",
        },
        inplace=True,
    )

    d_eo.drop(
        columns=["lat_bnd", "lon_bnd", "bnd", "std_dev", "numb"],
        inplace=True,
    )
    d_eo.rename(columns={"value": "satellite"}, inplace=True)

    joint_columns = list(set(d_gb.columns).intersection(set(d_eo.columns)))

    d_joint = d_gb.merge(d_eo, on=joint_columns, how="outer")

    return d_joint


@flow(
    name="combine_datasets",
)
def join_datasets_flow(path_to_csv: str, gas: str) -> None:
    """
    Run flow to combine satellite and ground-based data

    Parameters
    ----------
    path_to_csv:
        path where datasets are stored

    gas:
        target greenhouse gas variable
    """
    d_joint_wide = join_datasets_wide(path_to_csv=path_to_csv, gas=gas)
    d_joint_AK = apply_averaging_kernel(d_joint_wide)

    d_compare = d_joint_AK.dropna(subset="satellite").drop_duplicates()
    d_compare = (
        d_compare.groupby(
            ["time", "time_fractional", "year", "month", "lon", "lat", "gas", "unit"]
        )
        .agg(
            {
                "satellite": "mean",
                "ground_based": "mean",
                "ground_based_no_vertical": "mean",
                "ground_based_AK": "mean",
            }
        )
        .reset_index()
    )

    d_joint_long = joint_dataset_long(d_joint_AK)

    save_data(d_joint_AK, path_to_csv, gas, "joint_wide")
    save_data(d_compare, path_to_csv, gas, "joint_comparison")
    save_data(d_joint_long, path_to_csv, gas, "joint_long")


if __name__ == "__main__":
    join_datasets_flow("data/downloads", "co2")
