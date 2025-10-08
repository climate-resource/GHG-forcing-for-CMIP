"""
Test download data

Regression tests for downloading GHG concentrations
from web APIs
"""

import pandas as pd
import pytest

from ghg_forcing_for_cmip.download_ground_based import merge_netCDFs


@pytest.mark.parametrize("gas", ["co2", "ch4"])
@pytest.mark.parametrize("sampling_strategy", ["flask", "insitu"])
def test_download_extract_noaa(gas, sampling_strategy):
    if sampling_strategy == "insitu":
        expected_sites = ["brw", "mko"]
    else:
        expected_sites = ["abp", "alt", "ams"]

    # test merging
    df_all = []
    df_all.append(
        merge_netCDFs(
            extract_dir="tests/test_data/"
            + f"{gas}_surface-{sampling_strategy}_ccgg_netCDF"
        )
    )

    df_combined = pd.concat(df_all)

    expected_cols = [
        "std_dev",
        "numb",
        "value",
        "year",
        "month",
        "latitude",
        "longitude",
        "altitude",
        "site_code",
        "network",
        "insitu_vs_flask",
        "sampling_strategy",
        "gas",
        "unit",
        "version",
        "instrument",
    ]

    for col in expected_cols:
        assert col in df_combined.columns, f"Can't find column {col} in data frame."

    for code in expected_sites:
        assert (
            code.upper() in df_combined.site_code.unique()
        ), f"Can't finde site_code {code.upper()} in data frame."
