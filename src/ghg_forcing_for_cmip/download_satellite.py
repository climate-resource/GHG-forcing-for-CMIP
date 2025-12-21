"""
Download satellite data

In this module we download the OBS4MIPs data
used for satellite measurements of CO2 and CH4
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.datastores import Client
from prefect import flow, task

from ghg_forcing_for_cmip import CONFIG
from ghg_forcing_for_cmip.utils import (
    clean_and_save,
    unzip_download,
)
from ghg_forcing_for_cmip.validation import validate_eo_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(levelname)s: %(message)s",
)


@task(
    name="download_zip_from_cds",
    description="Download obs4mips zip from Climate Data Store",
    task_run_name="download_zip_from_cds_{gas}",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def make_api_request(gas: str, save_to_path: Path) -> None:
    """
    Request data from Climate Data Store via API

    Parameters
    ----------
    gas :
        target greenhouse gas variable
        Currently supported gas variables are 'co2' and 'ch4'

    save_to_path :
        path to save downloaded data

    References
    ----------
    Copernicus Climate Change Service, Climate Data Store, (2018):
    Methane data from 2002 to present derived from satellite observations.
    Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
    DOI: 10.24381/cds.b25419f8

    Copernicus Climate Change Service, Climate Data Store, (2018):
    "Carbon dioxide data from 2002 to present derived from satellite observations".
    Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
    DOI: 10.24381/cds.f74805c8

    Notes
    -----
    A login and further pre-setup is required to be
    able to download the data from the API.
    Please refer to the following link for further details:
    https://cds.climate.copernicus.eu/how-to-api
    """
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
    target = save_to_path / f"obs4mips_x{gas}.zip"

    client = Client(progress=False)

    if not client.check_authentication():
        raise ValueError("authentification of CDS client failed")  # noqa: TRY003

    client.retrieve(dataset, request, target=str(target))

    return logging.info(f"downloaded OBS4MIPs {gas} data to {target!s}")


@task(
    name="validate_obs4mips_data",
    description="validate obs4mips data",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def validate_obs4mips_data(
    path_to_nc: Path, gas: str = "co2", factor: float = 1e6
) -> pd.DataFrame:
    """
    Preprocess OBS4MIPS data from nc to csv format

    Parameters
    ----------
    path_to_nc:
        path where OBS4MIPS dataset is stored as nc format

    gas:
        target greenhouse gas variable

    factor:
        factor for converting ghg concentration from
        unitless to ppb (ch4; factor = 1e9) or
        ppm (co2; factor = 1e6)
    """
    all_files = os.listdir(path_to_nc)
    ds = next(
        file for file in all_files if "OBS4MIPS" in file and f"X{gas.upper()}" in file
    )

    df_raw = xr.open_dataset(path_to_nc / ds).to_dataframe().reset_index()
    df_raw = df_raw[df_raw[f"x{gas}"] != np.float32(1e20)].reset_index()
    df = pd.DataFrame({})

    df["time"] = pd.to_datetime(df_raw.time, utc=True)
    df["year"] = df_raw.time.dt.year.astype(np.int64)
    df["month"] = df_raw.time.dt.month.astype(np.int64)
    df["lat_bnd"] = df_raw.lat_bnds.astype(np.int64)
    df["lon_bnd"] = df_raw.lon_bnds.astype(np.int64)
    df["bnd"] = df_raw.bnds.astype(np.int64)
    df["lat"] = df_raw.lat.astype(np.float64)
    df["lon"] = df_raw.lon.astype(np.float64)
    df["value"] = df_raw[f"x{gas}"].astype(np.float64) * factor
    df["std_dev"] = df_raw[f"x{gas}_stddev"].astype(np.float64) * factor
    df["numb"] = df_raw[f"x{gas}_nobs"].astype(np.int64)
    df["gas"] = gas
    df["unit"] = np.where(gas == "ch4", "ppb", "ppm")
    df["pre"] = df_raw.pre.astype(np.float64)
    df["column_averaging_kernel"] = df_raw.column_averaging_kernel.astype(np.float64)
    df["vmr_profile_apriori"] = (
        df_raw[f"vmr_profile_{gas}_apriori"].astype(np.float64) * factor
    )

    validate_eo_dataframe(df)

    return df


@flow(name="download_satellite_data")
def download_satellite_data(
    gas: str, remove_original_files: bool, save_to_path: str = "data/downloads"
) -> None:
    """
    Download and preprocess satellite GHG concentration

    Parameters
    ----------
    gas :
        greenhouse gas,
        either ch4 or co2

    save_to_path :
        path to save the results

    remove_original_files :
        whether downloaded files should be kept;
        otherwise they are removed

    """
    save_to_path_arg = Path(save_to_path)

    make_api_request(gas=gas, save_to_path=save_to_path_arg)

    unzip_download.with_options(name="unzip_download")(
        zip_path=save_to_path_arg / f"obs4mips_x{gas}.zip",
        extract_dir=save_to_path_arg / f"{gas}/original",
    )

    df_final = validate_obs4mips_data(
        path_to_nc=save_to_path_arg / f"{gas}/original",
        gas=gas,
        factor=np.where(gas == "ch4", 1e9, 1e6),
    )

    # clean up repo and save file
    clean_and_save(
        df_final,
        gas=gas,
        save_to_path=save_to_path_arg,
        measurement_type="eo",
        remove_original_files=remove_original_files,
    )


if __name__ == "__main__":
    download_satellite_data(gas="co2", remove_original_files=False)
