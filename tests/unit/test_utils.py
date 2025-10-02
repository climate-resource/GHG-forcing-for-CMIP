"""
Test utils.py

Unit tests for helper functions
"""
import os
import pytest
import shutil
import pandas as pd

from ghg_forcing_for_cmip.utils import ensure_trailing_slash, clean_and_save


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
    clean_and_save(test_df, gas, "tests/test-data", "gb", False)

    # check if files are saved
    os.path.isfile(f"tests/test-data/{gas}/{gas}_gb_raw.csv")
    os.path.isfile(f"tests/test-data/{gas}/{gas}_gb_raw.nc")

    # remove test-folder after testing
    shutil.rmtree(f"tests/test-data/{gas}")