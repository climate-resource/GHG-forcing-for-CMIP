"""
Test download data

Regression tests for downloading GHG concentrations
from AGAGE web APIs
"""

from pathlib import Path

import pytest

from ghg_forcing_for_cmip.download_ground_based import download_agage, postprocess_agage


@pytest.mark.download
def test_download_agage():
    download_agage(Path("tests/test_results"))

    df_agage = postprocess_agage(
        Path("tests/test_results"), Path("tests/test_results/extracted/agage")
    )

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
        assert col in df_agage.columns, f"Can't find column {col} in data frame."

    # expected results
    with open(
        "tests/test_data/agage/expected_sites.txt",
        encoding="utf-8",
    ) as f:
        lines = f.read()
    expected_sites = lines.split()

    for code in expected_sites:
        assert (
            code.upper() in df_agage.site_code.unique()
        ), f"Can't finde site_code {code.upper()} in data frame."
