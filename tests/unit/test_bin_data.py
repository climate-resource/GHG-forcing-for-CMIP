"""
Unittest for binning observations in latitude x longitude grid
"""

import pandas as pd
import pytest

from ghg_forcing_for_cmip.bin_data import bin_minimum_grid


@pytest.fixture
def data_ch4_gb():
    return pd.read_csv("tests/test_data/ch4_gb_raw_test.csv")


def test_bin_dataset(data_ch4_gb):
    d_binned = bin_minimum_grid(data_ch4_gb)

    for var in [
        "time",
        "time_fractional",
        "bnd",
        "lon",
        "gas",
        "year",
        "month",
        "unit",
        "lat",
        "lat_bnd",
        "lon_bnd",
        "value_mean",
        "value_count",
    ]:
        assert var in d_binned.columns
