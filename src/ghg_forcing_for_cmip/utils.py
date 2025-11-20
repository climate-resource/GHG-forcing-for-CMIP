"""
Module including helper functions
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from prefect import task

from ghg_forcing_for_cmip import CONFIG

logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(levelname)s: %(message)s",
)


@task(
    name="clean_and_save",
    description="Clean directory and save file",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def clean_and_save(
    df: pd.DataFrame,
    gas: str,
    save_to_path: Path,
    measurement_type: str,
    remove_original_files: bool,
) -> None:
    """
    Clean up folder and save final dataset

    Parameters
    ----------
    df :
        final, post-processed data

    gas :
        greenhouse gas,
        either ch4 or co2

    save_to_path :
        path to save the results

    measurement_type :
        string used to discriminate ground_based
        and satellite data from each other;
        either "gb" or "eo"

    remove_original_files :
        whether downloaded files should be kept;
        otherwise they are removed
    """
    # save as csv
    df.to_csv(
        save_to_path / gas / f"{gas}_{measurement_type}_raw.csv",
        index=False,
    )

    # save as netcdf
    ds = xr.Dataset.from_dataframe(df)

    if measurement_type == "eo":
        lat, lon = "lat", "lon"
    else:
        lat, lon = "latitude", "longitude"

    year, month, latitude, longitude = (
        df[var].unique() for var in ["year", "month", lat, lon]
    )
    ds = ds.assign_coords({"year": year, "month": month, lat: latitude, lon: longitude})

    # reconvert datetime format (info is lost when converting df to ds)
    ds["time"] = df.time
    ds.to_netcdf(save_to_path / gas / f"{gas}_{measurement_type}_raw.nc")

    # clean-up directory
    if remove_original_files and os.path.exists(save_to_path / gas / "original"):
        shutil.rmtree(save_to_path / gas / "original")


@task(description="Unzip downloaded data", cache_policy=CONFIG.CACHE_POLICIES)
def unzip_download(zip_path: Path, extract_dir: Path) -> None:
    """
    Unzips a given ZIP file into the target directory.

    Parameters
    ----------
    zip_path :
        Path to the zip file (e.g., "data/downloads/noaa_ch4_surface_flask.zip")

    extract_dir :
        Path where the files should be extracted.
    """
    # make sure target directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # unzip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)

    logging.info(f"extracted {zip_path!s} to {extract_dir!s}")


def weighted_average(d: pd.DataFrame, grouping_vars: list[str]) -> pd.DataFrame:
    """
    Compute area weighted average

    Parameters
    ----------
    d :
        raw data frame

    grouping_vars :
        column names (variables) that
        should be used to average over

    Returns
    -------
    :
        data frame with value corresponding
        to weighted-by-area value
    """
    df = d.copy()
    cols = set(df.columns).difference(["bnd", "lat_bnd", "lon_bnd"])

    df_bnds = d.pivot_table(
        index=list(cols), columns="bnd", values=["lat_bnd", "lon_bnd"]
    ).reset_index()

    # flatten multi-index
    df_bnds.columns = pd.Index(
        ["_".join(str(c) for c in col if c != "") for col in df_bnds.columns.values]
    )

    # prepare computation of weighted average
    df_bnds["delta_lon"] = np.deg2rad(abs(df_bnds["lon_bnd_0"] - df_bnds["lon_bnd_1"]))
    df_bnds["delta_lat"] = abs(
        np.sin(abs(df_bnds["lat_bnd_0"])) - np.sin(abs(df_bnds["lat_bnd_1"]))
    )
    df_bnds["weight"] = df_bnds.delta_lon * df_bnds.delta_lat
    df_bnds["value_weighted"] = df_bnds.value * df_bnds.weight

    # group over relevant variables
    df_aggregated = (
        df_bnds.groupby(grouping_vars)
        .agg({"value_weighted": "sum", "weight": "sum"})
        .reset_index()
    )

    df_aggregated["value"] = df_aggregated.value_weighted / df_aggregated.weight

    # merge data sets column-wise
    df_aggregated.drop(columns=["value_weighted", "weight"], inplace=True)

    return df_aggregated
