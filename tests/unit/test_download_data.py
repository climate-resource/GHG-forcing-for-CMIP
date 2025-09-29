"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import numpy as np
import pandas as pd

from ghg_forcing_for_cmip.download_data import stats_from_events


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
