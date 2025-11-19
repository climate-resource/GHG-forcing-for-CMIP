"""
Test utils.py

Unit tests for helper functions
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ghg_forcing_for_cmip.utils import clean_and_save


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
