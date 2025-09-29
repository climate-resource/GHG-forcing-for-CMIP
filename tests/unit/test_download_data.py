"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.download_data import download_zip_from_noaa, stats_from_events


@pytest.mark.parametrize("gas", ["co2", "ch4"])
@pytest.mark.parametrize("sampling_strategy", ["flask", "insitu"])
def test_download_zip_from_noaa_real(gas, sampling_strategy, tmp_path):
    # Use temporary directory so no pollution
    save_dir = tmp_path

    # Run the actual function (downloads real NOAA data)
    download_zip_from_noaa(gas, sampling_strategy, save_to_path=str(save_dir))

    # Expected file name
    expected_file = save_dir / f"noaa_{gas}_surface_{sampling_strategy}.zip"

    # Assertions
    assert expected_file.exists(), f"Expected file {expected_file} was not created"
    assert expected_file.stat().st_size > 0, "Downloaded file is empty"


def test_stats_from_events():
    rng = np.random.default_rng(seed=123)
    expected_mean, expected_std, expected_count = (1.0, 0.8, 100_000)

    test_df = pd.DataFrame()
    test_df["value"] = rng.normal(expected_mean, expected_std, expected_count)
    test_df["time"] = pd.to_datetime("2018-01-01")
    for col in ["latitude", "longitude", "altitude"]:
        test_df[col] = 0

    observed_df = stats_from_events(test_df)

    np.testing.assert_almost_equal(observed_df.value.mean(), expected_mean, decimal=3)
    np.testing.assert_almost_equal(observed_df.std_dev.mean(), expected_std, decimal=3)
    np.testing.assert_equal(np.unique(observed_df.numb), expected_count)
