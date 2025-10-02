"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import numpy as np
import pandas as pd

from ghg_forcing_for_cmip.download_ground_based import (
    add_lat_lon_bnds,
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


def test_add_lat_lon_bnds():
    df_test = pd.DataFrame()
    df_test["idx"] = [0, 1, 2, 3, 4, 5]
    df_test["latitude"] = [2.5, 12.4, -90, -27.3, 90, 0]
    df_test["longitude"] = [-6.5, -180, 45.5, 180, 2.5, 0]

    df_lat_lon = add_lat_lon_bnds(df_test)

    expected_lower_lon = [-5, -175, 45, 175, 0, -5]
    expected_upper_lon = [-10, -180, 50, 180, 5, 0]
    expected_lower_lat = [0, 10, -85, -25, 85, -5]
    expected_upper_lat = [5, 15, -90, -30, 90, 0]

    np.testing.assert_array_equal(
        df_lat_lon[df_lat_lon.bnd == 0]["lat_bnd"].values, expected_lower_lat
    )
    np.testing.assert_array_equal(
        df_lat_lon[df_lat_lon.bnd == 0]["lon_bnd"].values, expected_lower_lon
    )
    np.testing.assert_array_equal(
        df_lat_lon[df_lat_lon.bnd == 1]["lat_bnd"].values, expected_upper_lat
    )
    np.testing.assert_array_equal(
        df_lat_lon[df_lat_lon.bnd == 1]["lon_bnd"].values, expected_upper_lon
    )


def test_get_indices():
    input_and_expected = [0, 4.0, 5.0, 6.0, 9, 17]
    observed = get_indices(input_and_expected, np.arange(18))

    np.testing.assert_array_equal(observed, input_and_expected)
