"""
Test download data

Regression tests for downloading GHG concentrations
from web APIs
"""

import os

import numpy as np
import pytest

from ghg_forcing_for_cmip.download_data import download_zip_from_noaa, unzip_download


@pytest.mark.parametrize("gas", ["co2", "ch4"])
@pytest.mark.parametrize("sampling_strategy", ["flask", "insitu"])
def test_download_extract_noaa(gas, sampling_strategy):
    # expected results
    with open(
        f"tests/expected_noaa/expected_sites_{gas}_{sampling_strategy}.txt",
        encoding="utf-8",
    ) as f:
        lines = f.read()
    expected_sites = lines.split()

    # Note: number of file plus Readme
    exp_numb_files = len(expected_sites) + 1

    save_dir = "tests/test-data/"

    # download NOAA data in test directory
    download_zip_from_noaa(gas, sampling_strategy, save_to_path=save_dir)

    # Expected file name
    expected_file = save_dir + f"noaa_{gas}_surface_{sampling_strategy}.zip"

    # Assertions for downloading zip
    assert os.path.isfile(
        expected_file
    ), f"Expected file {expected_file} was not created"
    assert os.stat(expected_file).st_size > 0, "Downloaded file is empty"

    # unzip folder
    unzip_download(zip_path=expected_file, extract_dir=save_dir + "extracted")

    all_files = os.listdir(
        save_dir + f"extracted/{gas}_surface-{sampling_strategy}_ccgg_netCDF"
    )

    observed_sites = [
        site.split("_")[1] for site in all_files if not site.startswith("README")
    ]

    # check that number of files is as expected
    assert len(all_files) == exp_numb_files
    # check that all observation sites are included as expected
    np.testing.assert_array_equal(np.sort(expected_sites), np.sort(observed_sites))
