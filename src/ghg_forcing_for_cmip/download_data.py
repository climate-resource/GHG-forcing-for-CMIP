"""
Download data

The task of this module is data scraping of the GHG
concentration data from e.g. (A)GAGE, NOAA
"""

import os

import requests
from prefect import task

from ghg_forcing_for_cmip import utils


@task(
    name="download_zip_from_noaa_archive",
    description="Download zip from NOAA archive link",
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
