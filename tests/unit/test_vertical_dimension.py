"""
Unittests for vertical dimension
"""

import os

import pandas as pd
import pytest

from ghg_forcing_for_cmip.vertical_dimension import method_meinshausen2017


@pytest.fixture
def gb_ch4_data():
    print(os.getcwd())
    return pd.read_csv("data/downloads/ch4/ch4_gb_raw.csv")


def test_add_vertical(gb_ch4_data):
    method_meinshausen2017(gb_ch4_data)
