"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import numpy as np
import pytest

from ghg_forcing_for_cmip.download_data import download_zip_from_noaa


@pytest.mark.parametrize(
    "gas,sampling_strategy",
    [
        ("co2", "in-situ"),
        ("ch4", "flask"),
    ],
)
def test_download_zip_from_noaa_real(gas, sampling_strategy, tmp_path):
    # Use temporary directory so no pollution
    save_dir = tmp_path

    # Run the actual function (downloads real NOAA data)
    download_zip_from_noaa(gas, sampling_strategy, save_to_path=str(save_dir))

    # Expected file name
    strat_name = "insitu" if sampling_strategy == "in-situ" else "flask"
    expected_file = save_dir / f"noaa_{gas}_surface_{strat_name}.zip"

    # Assertions
    np.testing.assert_(
        expected_file.exists(), f"Expected file {expected_file} was not created"
    )
    np.testing.assert_(expected_file.stat().st_size > 0, "Downloaded file is empty")
