"""
Bin dataset into latitude x longitude grid
"""

from typing import Optional

import pandas as pd
from prefect import flow, task

from ghg_forcing_for_cmip import CONFIG


@task(description="Bin data into 5x5 grid", cache_policy=CONFIG.CACHE_POLICIES)
def bin_minimum_grid(data: pd.DataFrame):
    """
    Bin dataset into 5 x 5 grid

    Parameters
    ----------
    data :
        raw data set

    Returns
    -------
    :
        aggregated / binned data set
    """
    cols = set(data.columns).difference(
        [
            "latitude",
            "longitude",
            "instrument",
            "network",
            "std_dev",
            "insitu_vs_flask",
            "site_code",
            "numb",
            "altitude",
            "version",
            "value",
            "sampling_strategy",
        ]
    )

    d_aggregated = (
        data.groupby(list(cols)).agg({"value": ["mean", "count"]}).reset_index()
    )

    # flatten multi-index
    d_aggregated.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in d_aggregated.columns.to_flat_index()
    ]

    d_aggregated.rename(
        columns={"value_mean": "value", "value_count": "n_value"}, inplace=True
    )

    return d_aggregated


@flow(name="bin data")
def bin_data(load_from_path: str, save_to_path: Optional[str]) -> None:
    """
    Bin data set

    Parameters
    ----------
    load_from_path :
        path from where to load the raw data set

    save_to_path :
        path where to save the binned data set
        if None, dataset is not saved
    """
    data = pd.read_csv(load_from_path)

    d_binned = bin_minimum_grid(data)

    # d_binned_agg = weighted_average(d_binned, ["lon", "year", "month"])

    if save_to_path is not None:
        d_binned.to_csv(save_to_path)


if __name__ == "__main__":
    bin_data(load_from_path="data/downloads/ch4/ch4_gb_raw.csv", save_to_path=None)
