"""
Test utils.py

Unit tests for helper functions
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.utils import clean_and_save, weighted_average


@pytest.fixture
def data_ch4_gb():
    return pd.read_csv("tests/test_data/ch4_gb_raw_test.csv")


@pytest.mark.parametrize("measurement_type", ["gb", "eo"])
@pytest.mark.parametrize("gas", ["ch4", "co2"])
def test_clean_and_save(gas, measurement_type):
    path_to_results = Path("tests/test_results")

    # prepare test dataframe
    test_df = pd.DataFrame()
    for var, val in zip(
        ["year", "month", "latitude", "longitude"], [2015, 1, 2.5, 2.5]
    ):
        test_df[var] = val

    if measurement_type == "eo":
        test_df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

    test_df["time"] = pd.to_datetime("01.02.2028")

    # create folder if not existent
    os.makedirs(path_to_results / gas, exist_ok=True)

    # run target function
    clean_and_save(test_df, gas, path_to_results, measurement_type, False)

    # check if files are saved
    os.path.isfile(path_to_results / f"{gas}/{gas}_{measurement_type}_raw.csv")
    os.path.isfile(path_to_results / f"{gas}/{gas}_{measurement_type}_raw.nc")

    # remove test-folder after testing
    shutil.rmtree(path_to_results / gas)


def test_weighted_average(data_ch4_gb):
    # average over bnd=0 and bnd=1 conditions(= cell mean)
    agg_cols = ["longitude", "latitude", "lon", "lat", "year", "month"]

    d_test = data_ch4_gb.groupby(agg_cols).agg({"value": "mean"}).reset_index()

    d_weighted = weighted_average(data_ch4_gb, agg_cols)

    d_merged = d_weighted.merge(
        d_test, how="outer", on=agg_cols, suffixes=("_agg", "_orig")
    )

    # compare that cell means are the same (we don't average over latitudes,
    # thus no further modification)
    np.testing.assert_allclose(d_merged.value_agg, d_merged.value_orig)

    # average over latitudes
    agg_cols2 = ["lon", "year", "month"]

    d_test2 = data_ch4_gb.groupby(agg_cols2).agg({"value": "mean"}).reset_index()

    d_weighted2 = weighted_average(data_ch4_gb, agg_cols2)

    d_merged2 = d_weighted2.merge(
        d_test2, how="outer", on=agg_cols2, suffixes=("_agg", "_orig")
    )

    # this should fail, as its not only a simple average but weighting takes
    # shape transformation of grid-cells into account
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(d_merged2.value_agg, d_merged2.value_orig)
