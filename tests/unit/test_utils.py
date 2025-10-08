"""
Test utils.py

Unit tests for helper functions
"""

import os
import shutil

import pandas as pd
import pytest

from ghg_forcing_for_cmip.utils import clean_and_save, ensure_trailing_slash


@pytest.mark.parametrize(
    "test_path, expected",
    [
        ("test_path_without", "test_path_without/"),
        ("test_path_with/", "test_path_with/"),
        (".", "./"),
        ("/", "/"),
    ],
)
def test_ensure_trailing_slash(test_path, expected):
    observed = ensure_trailing_slash(test_path)

    assert observed == expected, (
        f"The observed test-path: {observed},"
        " does not match the expected path: {expected}"
    )


@pytest.mark.parametrize("measurement_type", ["gb", "eo"])
@pytest.mark.parametrize("gas", ["ch4", "co2"])
def test_clean_and_save(gas, measurement_type):
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
    os.makedirs(f"tests/test_results/{gas}", exist_ok=True)

    # run target function
    clean_and_save(test_df, gas, "tests/test_results", measurement_type, False)

    # check if files are saved
    os.path.isfile(f"tests/test_results/{gas}/{gas}_{measurement_type}_raw.csv")
    os.path.isfile(f"tests/test_results/{gas}/{gas}_{measurement_type}_raw.nc")

    # remove test-folder after testing
    shutil.rmtree(f"tests/test_results/{gas}")
