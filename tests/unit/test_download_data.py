"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.download_data import clean_and_save, stats_from_events


def test_stats_from_events():
    rng = np.random.default_rng(seed=123)
    expected_mean, expected_std, expected_count = (1.0, 0.8, 100_000)

    test_df = pd.DataFrame()
    test_df["value"] = rng.normal(expected_mean, expected_std, expected_count)
    test_df["time"] = pd.to_datetime("2018-01-01")
    for col in ["latitude", "longitude", "altitude"]:
        test_df[col] = 0

    observed_df = stats_from_events(test_df)

    np.testing.assert_almost_equal(observed_df.value.mean(), expected_mean, decimal=3)
    np.testing.assert_almost_equal(observed_df.std_dev.mean(), expected_std, decimal=3)
    np.testing.assert_equal(np.unique(observed_df.numb), expected_count)


@pytest.mark.parametrize("gas", ["ch4", "co2"])
def test_clean_and_save(gas):
    # prepare test dataframe
    test_df = pd.DataFrame()
    for var, val in zip(
        ["year", "month", "latitude", "longitude"], [2015, 1, 2.5, 2.5]
    ):
        test_df[var] = val
    test_df["time"] = pd.to_datetime("01.02.2028")

    # create folder if not existent
    os.makedirs(f"tests/test-data/{gas}", exist_ok=True)

    # run target function
    clean_and_save(test_df, gas, "tests/test-data", False)

    # check if files are saved
    os.path.isfile(f"tests/test-data/{gas}/{gas}_raw.csv")
    os.path.isfile(f"tests/test-data/{gas}/{gas}_raw.nc")

    # remove test-folder after testing
    shutil.rmtree(f"tests/test-data/{gas}")
