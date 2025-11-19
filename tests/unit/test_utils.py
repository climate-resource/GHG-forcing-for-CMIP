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


@pytest.fixture
def gb_ch4_data():
    return pd.read_csv("data/downloads/ch4/ch4_gb_raw.csv")


def test_weighted_average(gb_ch4_data):
    d_weighted = weighted_average(
        gb_ch4_data, ["longitude", "latitude", "lon", "lat", "year", "month"]
    )

    np.testing.assert_allclose(d_weighted.value_orig, d_weighted.value_weighted)
