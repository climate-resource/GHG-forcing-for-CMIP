"""
test downloading of datasets for ghg comparison
"""

import os

import numpy as np
import pandas as pd
import pytest

from ghg_forcing_for_cmip.data_comparison.get_datasets import (
    add_lat_lon_bnds,
    combine_netCDFs,
    download_agage,
    download_noaa,
    unzip_download,
)


def test_download_agage() -> None:
    download_agage_uncached = download_agage.with_options(cache_key_fn=lambda *_: None)
    # check whether folder has been created
    df = download_agage_uncached(
        path_to_source="data/downloads/ch4/agage/agage_data",
    )

    cols = df.columns
    expected_cols = [
        "latitude",
        "longitude",
        "site_code",
        "network",
        "altitude",
        "instrument",
        "version",
        "gas",
        "value",
        "std_dev",
        "numb",
        "time",
        "year",
        "month",
        "unit",
        "insitu_vs_flask",
        "sampling_strategy",
    ]

    np.testing.assert_equal(sorted(cols), sorted(expected_cols))


@pytest.mark.long
def test_download_noaa() -> None:
    download_noaa_uncached = download_noaa.with_options(cache_key_fn=lambda *_: None)
    # check whether folder has been created
    for sampling in ["in-situ", "flask"]:
        download_noaa_uncached(
            gas="ch4",
            sampling_strategy=sampling,
            save_to_path="tests/test-data/download_noaa/",
        )

    files = os.listdir("tests/test-data/download_noaa/")
    obs_files = [file for file in files if file.endswith(".zip")]
    expected_files = ["noaa_ch4_surface_flask.zip", "noaa_ch4_surface_insitu.zip"]

    np.testing.assert_equal(sorted(obs_files), sorted(expected_files))


@pytest.mark.parametrize("sampling_strategy", ["insitu", "flask"])
def test_unzip_download(sampling_strategy) -> None:
    gas = "ch4"
    unzip_download_uncached = unzip_download.with_options(cache_key_fn=lambda *_: None)
    unzip_download_uncached(
        path_to_zip=f"tests/test-data/download_noaa/noaa_ch4_surface_{sampling_strategy}.zip",
        zip_folder=f"{gas}/noaa/{gas}_surface-insitu_ccgg_netCDF",
        save_to="tests/test-data/download_noaa",
    )

    np.testing.assert_(
        os.path.isdir(
            f"tests/test-data/download_noaa/ch4_surface-{sampling_strategy}_ccgg_netCDF"
        )
    )
    np.testing.assert_(
        not os.path.isfile(
            f"test/test-data/download_noaa/noaa_ch4_surface_{sampling_strategy}.zip"
        )
    )


@pytest.mark.parametrize("sampling_strategy", ["insitu", "flask"])
def test_combine_netCDFs(sampling_strategy) -> None:
    combine_netCDFs_uncached = combine_netCDFs.with_options(
        cache_key_fn=lambda *_: None
    )
    df = combine_netCDFs_uncached(
        path_to_files=f"tests/test-data/download_noaa/ch4_surface-{sampling_strategy}_ccgg_netCDF"
    )

    expected_cols = [
        "time",
        "year",
        "month",
        "latitude",
        "longitude",
        "value",
        "site_code",
        "network",
        "altitude",
        "insitu_vs_flask",
        "sampling_strategy",
        "gas",
        "unit",
        "std_dev",
        "numb",
        "instrument",
        "version",
    ]

    np.testing.assert_equal(sorted(df.columns), sorted(expected_cols))


@pytest.mark.parametrize("sampling_strategy", ["insitu", "flask"])
def test_add_lat_lon(sampling_strategy) -> None:
    combine_netCDFs_uncached = combine_netCDFs.with_options(
        cache_key_fn=lambda *_: None
    )
    add_lat_lon_uncached = add_lat_lon_bnds.with_options(cache_key_fn=lambda *_: None)

    df = combine_netCDFs_uncached(
        path_to_files=f"tests/test-data/download_noaa/ch4_surface-{sampling_strategy}_ccgg_netCDF"
    )

    df2 = add_lat_lon_uncached(df)

    expected_cols = [
        "time",
        "year",
        "month",
        "latitude",
        "longitude",
        "value",
        "site_code",
        "network",
        "altitude",
        "insitu_vs_flask",
        "sampling_strategy",
        "gas",
        "unit",
        "std_dev",
        "numb",
        "lat",
        "lon",
        "bnd",
        "lat_bnd",
        "lon_bnd",
        "instrument",
        "version",
    ]

    np.testing.assert_equal(sorted(df2.columns), sorted(expected_cols))


def test_combine_dataframes(gas="ch4"):
    combine_netCDFs_uncached = combine_netCDFs.with_options(
        cache_key_fn=lambda *_: None
    )
    download_agage_uncached = download_agage.with_options(cache_key_fn=lambda *_: None)

    df_insitu = combine_netCDFs_uncached(
        f"tests/test-data/download_noaa/{gas}_surface-insitu_ccgg_netCDF"
    )
    df_flask = combine_netCDFs_uncached(
        f"tests/test-data/download_noaa/{gas}_surface-flask_ccgg_netCDF"
    )
    # check whether folder has been created
    if gas != "ch4":
        df_agage = None
    else:
        df_agage = download_agage_uncached(
            path_to_source="data/downloads/ch4/agage/agage_data",
        )

    df_combined = pd.concat([df_insitu, df_flask, df_agage])

    np.testing.assert_equal(
        len(df_combined), len(df_insitu) + len(df_flask) + len(df_agage)
    )
