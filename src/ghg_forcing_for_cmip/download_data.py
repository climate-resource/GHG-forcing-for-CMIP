"""
Download data

The task of this module is data scraping of the GHG
concentration data from e.g. (A)GAGE, NOAA
"""

import os
import zipfile

import requests
from prefect import task

from ghg_forcing_for_cmip import utils


@task(
    name="download_noaa_zip",
    description="Download zip-folder from NOAA",
    refresh_cache=True,
    persist_result=False,
)
def download_zip_from_noaa(
    gas: str,
    sampling_strategy: str,
    save_to_path: str = "data/downloads",
) -> None:
    """
    Download NOAA data as NETCDF zip-file

    Parameters
    ----------
    gas :
        target greenhouse gas variable;
        either 'co2' or 'ch4'

    sampling_strategy :
        either 'in-situ' or 'flask'

    save_to_path :
        path to save downloaded data
    """
    # setup directory
    save_to_path = utils.ensure_trailing_slash(save_to_path)
    os.makedirs(save_to_path, exist_ok=True)

    if sampling_strategy == "in-situ":
        dir, sampling_strategy = "in-situ", "insitu"
    if sampling_strategy == "flask":
        dir, sampling_strategy = "flask", "flask"

    url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{dir}/surface/{gas}_surface-{sampling_strategy}_ccgg_netCDF.zip"

    # note: probably timeout has to be adjusted (currently only an initial guess)
    response = requests.get(url, timeout=10)

    with open(save_to_path + f"/noaa_{gas}_surface_{sampling_strategy}.zip", "wb") as f:
        f.write(response.content)

    print(f"downloaded NOAA-zip ({gas}-{sampling_strategy}) to {save_to_path}")


@task(
    name="unzip_download",
    description="Unzip downloaded data",
    refresh_cache=True,
    persist_result=False,
)
def unzip_download(zip_path: str, extract_dir: str) -> None:
    """
    Unzips a given ZIP file into the target directory (default: data/downloads/ch4).

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

    print(f"Extracted {zip_path} to {extract_dir}")


if __name__ == "__main__":
    # download_zip_from_noaa("ch4", "flask")
    unzip_download("data/downloads/noaa_ch4_surface_flask.zip", "data/downloads/ch4")
