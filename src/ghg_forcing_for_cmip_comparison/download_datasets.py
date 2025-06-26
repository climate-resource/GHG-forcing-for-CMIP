"""
Download obs4mips and cmip data sets
"""

import os
import re
import zipfile

import cdsapi  # type: ignore
from prefect import flow, task

from ghg_forcing_for_cmip_comparison.utils import custom_cache_key_fn


@task(
    name="download_obs4mips_data",
    description="Download obs4mips data from Climate Data Store",
    task_run_name="download_obs4mips_{gas}_data",
    cache_key_fn=custom_cache_key_fn(),
)
def make_api_request(gas: str, save_to_path: str = "data/downloads") -> None:
    """
    Request data from Climate Data Store via API

    A login and further pre-setup is required to be
    able to download the data from the API.
    Please refer to the following link for further details:
    https://cds.climate.copernicus.eu/how-to-api

    Parameters
    ----------
    gas :
        target greenhouse gas variable
        Currently supported gas variables are 'co2' and 'ch4'

    save_to_path :
        path to save downloaded data
    """
    # TODO: Is there a way to check whether the /.cdsapirc file is
    #  somewhere on the computer?

    if gas == "co2":
        dataset = "satellite-carbon-dioxide"
    elif gas == "ch4":
        dataset = "satellite-methane"
    else:
        raise ValueError(f"gas must be 'ch4', 'co2'. Got {gas=}")  # noqa: TRY003

    # create directory if it doesn't exist
    os.makedirs(save_to_path, exist_ok=True)

    request = {
        "processing_level": ["level_3"],
        "variable": f"x{gas}",
        "sensor_and_algorithm": "merged_obs4mips",
        "version": ["4_5"],
    }
    # setup saving location
    target = save_to_path + f"/obs4mips_x{gas}_v{request['version'][0]}.zip"

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target=target)

    return print(f"downloaded data to {target}")


@task(
    name="unzip_downloaded_obs4mips_data",
    description="Unzip downloaded obs4mips data",
    cache_key_fn=custom_cache_key_fn(),
)
def unzip_obs4mips_download(
    gas: str, path_to_obs4mips_zip: str, path_to_obs4mips_nc: str
) -> None:
    """
    Unzip downloaded obs4mips data

    Parameters
    ----------
    gas :
        target greenhouse gas variable

    path_to_obs4mips_zip :
        path to downloaded obs4mips data

    path_to_obs4mips_nc :
        path to unzipped obs4mips data file (in netCDF format)

    """
    files = os.listdir(path_to_obs4mips_zip)
    pattern = re.compile(f"obs4mips_x{gas}")

    # Find the first matching file or raise an error if none found
    target_file = next((f for f in files if pattern.match(f)), None)
    if target_file is None:
        raise FileNotFoundError(  # noqa: TRY003
            f"No file matching pattern obs4mips_x{gas}"
            f" found in {path_to_obs4mips_zip}"
        )

    zip_path = os.path.join(path_to_obs4mips_zip, target_file)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path_to_obs4mips_nc)

    return print(f"unzipped data to {path_to_obs4mips_nc}")


@flow(name="download_datasets", description="Download CMIP and OBS4MIPs data")
def download_datasets_flow(save_to_path: str = "data/downloads") -> None:
    """
    Download and extract CMIP7 and OBS4MIPs data

    Parameters
    ----------
    save_to_path :
        path to save downloaded data
    """
    for gas in ("co2", "ch4"):
        make_api_request(gas=gas, save_to_path=save_to_path)
        unzip_obs4mips_download(
            gas=gas, path_to_obs4mips_zip=save_to_path, path_to_obs4mips_nc=save_to_path
        )


if __name__ == "__main__":
    download_datasets_flow()
