"""
Test download data

Regression tests for downloading GHG concentrations
from web APIs
"""

import os

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.download_ground_based import (
    download_zip_from_noaa,
    merge_netCDFs,
)
from ghg_forcing_for_cmip.utils import unzip_download


@pytest.mark.parametrize("gas", ["co2", "ch4"])
@pytest.mark.parametrize("sampling_strategy", ["flask"])  # "insitu"
def test_download_extract_noaa(gas, sampling_strategy):
    # expected results
    with open(
        f"test_download_data/expected_noaa/expected_sites_{gas}_{sampling_strategy}.txt",
        encoding="utf-8",
    ) as f:
        lines = f.read()
    expected_sites = lines.split()

    # Note: number of file plus Readme
    exp_numb_files = len(expected_sites) + 1

    save_dir = "test_download_data/test-data/"

    # %% test downloading
    # download NOAA data in test directory
    download_zip_from_noaa(gas, sampling_strategy, save_to_path=save_dir)

    # Expected file name
    expected_file = save_dir + f"noaa_{gas}_surface_{sampling_strategy}.zip"

    # Assertions for downloading zip
    assert os.path.isfile(
        expected_file
    ), f"Expected file {expected_file} was not created"
    assert os.stat(expected_file).st_size > 0, "Downloaded file is empty"

    # %% test unzipping of files
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

    # %% test merging
    df_all = []
    df_all.append(
        merge_netCDFs(
            extract_dir=save_dir
            + f"extracted/{gas}_surface-{sampling_strategy}_ccgg_netCDF"
        )
    )

    df_combined = pd.concat(df_all)

    expected_cols = [
        "std_dev",
        "numb",
        "value",
        "year",
        "month",
        "latitude",
        "longitude",
        "altitude",
        "site_code",
        "network",
        "insitu_vs_flask",
        "sampling_strategy",
        "gas",
        "unit",
        "version",
        "instrument",
    ]

    for col in expected_cols:
        assert col in df_combined.columns, f"Can't find column {col} in data frame."

    for code in expected_sites:
        assert (
            code.upper() in df_combined.site_code.unique()
        ), f"Can't finde site_code {code.upper()} in data frame."
