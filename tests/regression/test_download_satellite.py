"""
Test download of satellite data

Regression tests for processing satellite data
"""

import os

import numpy as np
import pytest

from ghg_forcing_for_cmip.download_satellite import validate_obs4mips_data
from ghg_forcing_for_cmip.utils import (
    clean_and_save,
    ensure_trailing_slash,
)


@pytest.mark.parametrize("gas", ["ch4", "co2"])
def test_download_satellite_data(gas, save_to_path="tests/test-data/satellite"):
    os.makedirs(save_to_path + "/" + gas, exist_ok=True)

    save_to_path = ensure_trailing_slash(save_to_path)

    df_final = validate_obs4mips_data(
        path_to_nc="tests/expected_data/",
        gas=gas,
        factor=np.where(gas == "ch4", 1e9, 1e6),
    )

    assert df_final.year.unique() == 2014
    assert df_final["time"].dtype == "datetime64[ns, UTC]"

    clean_and_save(
        df_final,
        gas=gas,
        save_to_path=save_to_path,
        measurement_type="eo",
        remove_original_files=True,
    )
