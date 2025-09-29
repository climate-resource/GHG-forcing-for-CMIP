"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.download_data import (
    add_lat_lon_bnds,
    clean_and_save,
    get_indices,
    stats_from_events,
)


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
    clean_and_save(test_df, gas, "tests/test-data", False)

    # check if files are saved
    os.path.isfile(f"tests/test-data/{gas}/{gas}_raw.csv")
    os.path.isfile(f"tests/test-data/{gas}/{gas}_raw.nc")

    # remove test-folder after testing
    shutil.rmtree(f"tests/test-data/{gas}")


@pytest.mark.parametrize(
    "lat, expected_lower_lat, expected_upper_lat, "
    "lon, expected_lower_lon, expected_upper_lon",
    [
        (2.5, 0, 5, -6.5, -5, -10.0),
        (12.4, 10, 15, -180, -175, -180),
        (-90, -85, -90, 45.5, 45, 50),
        (-27.3, -25, -30, 179, 175, 180),
        (90, 85, 90, 2.5, 0, 5),
        (0, -5, 0, 0, -5, 0),
    ],
)
def test_add_lat_lon_bnds(  # noqa: PLR0913
    lat,
    expected_lower_lat,
    expected_upper_lat,
    lon,
    expected_lower_lon,
    expected_upper_lon,
):
    df_test = pd.DataFrame()
    df_test["latitude"] = [lat]
    df_test["longitude"] = [lon]

    df_lat_lon = add_lat_lon_bnds.with_options(cache_expiration=0.0)(df_test)

    for band, band_val in zip(["lower", "upper"], [0, 1]):
        observed_lat_bnd = df_lat_lon[df_lat_lon.bnd == band_val]["lat_bnd"].values[0]
        assert observed_lat_bnd == np.where(
            band == "lower", expected_lower_lat, expected_upper_lat
        ), (
            f"For lat={lat} observed {band} bound ({observed_lat_bnd})"
            + f" is not equal to expected {band} bound "
            + f"({np.where(band == 'lower', expected_lower_lat, expected_upper_lat)})"
        )

        observed_lon_bnd = df_lat_lon[df_lat_lon.bnd == band_val]["lon_bnd"].values[0]
        assert observed_lon_bnd == np.where(
            band == "lower", expected_lower_lon, expected_upper_lon
        ), (
            f"For lon={lon} observed {band} bound ({observed_lon_bnd})"
            + f" is not equal to expected {band} bound"
            + f"({np.where(band == 'lower', expected_lower_lon, expected_upper_lon)})"
        )


def test_get_indices():
    input_and_expected = [0, 4.0, 5.0, 6.0, 9, 17]
    observed = get_indices(input_and_expected, np.arange(18))

    np.testing.assert_array_equal(observed, input_and_expected)
