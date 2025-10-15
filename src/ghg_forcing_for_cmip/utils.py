"""
Module including helper functions
"""

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr
from prefect import task
from tqdm import tqdm

from ghg_forcing_for_cmip import CONFIG
from ghg_forcing_for_cmip.download_ground_based import download_surface_data
from ghg_forcing_for_cmip.download_satellite import download_satellite_data


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

    print(f"Extracted {zip_path!s} to {extract_dir!s}")


def download_data():
    """
    Download satellite and ground-based data

    used for tutorials to download data when running docs
    """
    for gas in tqdm(["ch4", "co2"]):
        download_surface_data(gas=gas, remove_original_files=True)
        download_satellite_data(gas=gas, remove_original_files=True)
