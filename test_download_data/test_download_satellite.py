"""
Test download of satellite data

Regression tests for downloading GHG concentrations
from climate data store
"""

import os
from pathlib import Path

import numpy as np
import pytest

from ghg_forcing_for_cmip.download_satellite import (
    make_api_request,
    validate_obs4mips_data,
)
from ghg_forcing_for_cmip.utils import (
    clean_and_save,
    unzip_download,
)


@pytest.mark.parametrize("gas", ["ch4", "co2"])
def test_download_satellite_data(
    gas, save_to_path=Path("tests/test_results/satellite")
):
    make_api_request(gas, save_to_path)

    assert os.path.exists(save_to_path / f"obs4mips_x{gas}.zip"), (
        f"Can't find obs4mips_x{gas}.zip in " + str(save_to_path)
    )

    unzip_download(
        zip_path=save_to_path / f"obs4mips_x{gas}.zip",
        extract_dir=save_to_path / f"{gas}/original",
    )

    assert os.path.isdir(save_to_path / f"{gas}/original"), (
        "Can't find directory: " + f"{gas}/original"
    )
    assert os.listdir(save_to_path / f"{gas}/original"), (
        "Directory " + f"{gas}/original" + " is empty."
    )

    df_final = validate_obs4mips_data(
        path_to_nc=save_to_path / f"{gas}/original",
        gas=gas,
        factor=np.where(gas == "ch4", 1e9, 1e6),
    )

    np.testing.assert_array_equal(df_final.year.unique(), np.arange(2003, 2023))
    assert df_final["time"].dtype == "datetime64[ns, UTC]"

    clean_and_save(
        df_final,
        gas=gas,
        save_to_path=save_to_path,
        measurement_type="eo",
        remove_original_files=True,
    )
