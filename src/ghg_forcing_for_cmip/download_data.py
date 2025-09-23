"""
Download data

The task of this module is data scraping of the GHG
concentration data from e.g. (A)GAGE, NOAA
"""

import os
import zipfile

import numpy as np
import requests
import xarray as xr
from prefect import flow, task

from ghg_forcing_for_cmip import utils


@task(
    name="download_noaa_zip",
    description="Download zip-folder from NOAA",
)
def download_zip_from_noaa(gas: str, sampling_strategy: str, save_to_path: str) -> None:
    """
    Download NOAA data as NETCDF zip-file

    Parameters
    ----------
    gas :
        target greenhouse gas variable;
        either 'co2' or 'ch4'

    sampling_strategy :
        either 'insitu' or 'flask'

    save_to_path :
        path to save downloaded data
    """
    # setup directory
    save_to_path = utils.ensure_trailing_slash(save_to_path)
    os.makedirs(save_to_path, exist_ok=True)

    if sampling_strategy == "insitu":
        folder, sampling_strategy = "in-situ", "insitu"
    if sampling_strategy == "flask":
        folder, sampling_strategy = "flask", "flask"

    url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{folder}/surface/{gas}_surface-{sampling_strategy}_ccgg_netCDF.zip"

    # note: probably timeout has to be adjusted (currently only an initial guess)
    response = requests.get(url, timeout=10)

    with open(save_to_path + f"/noaa_{gas}_surface_{sampling_strategy}.zip", "wb") as f:
        f.write(response.content)

    print(f"downloaded NOAA-zip ({gas}-{sampling_strategy}) to {save_to_path}")


@task(
    name="unzip_download",
    description="Unzip downloaded data",
)
def unzip_download(zip_path: str, extract_dir: str) -> None:
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

    print(f"Extracted {zip_path} to {extract_dir}")


def stats_from_events(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute summary statistics from event-data

    Compute mean, std_dev, and count values across
    events for insitu data.

    Parameters
    ----------
    ds :
        insitu dataset

    Returns
    -------
    :
        dataset including mean value, std, and count
        across events for insitu data
    """
    df = ds.to_dataframe().reset_index()
    cols = list(df.columns)

    for item in ["time", "value"]:
        cols.remove(item)
    df = df.groupby(cols).agg({"value": ["mean", "std", "count"]}).reset_index()

    df.columns = [
        "_".join(map(str, col)).strip("_")
        for col in list(df.columns.values)  # type: ignore
    ]
    df.rename(
        columns={
            "value_mean": "value",
            "value_std": "std_dev",
            "value_count": "numb",
        },
        inplace=True,
    )

    ds = xr.Dataset.from_dataframe(df)

    return ds[["value", "std_dev", "numb"]]


@task(
    name="merge_netCDFs",
    description="Merge information from single files into one single netCDF",
)
def merge_netCDFs(
    extract_dir: str,
) -> xr.Dataset:
    """
    Combine netCDF files into a single dataframe

    Parameters
    ----------
    extract_dir :
        path to extracted netCDF files

    Returns
    -------
    :
        dataset combining all single datafiles
    """
    files = os.listdir(extract_dir)
    nc_files = [file for file in files if file.endswith(".nc")]

    df_list = []
    for file in nc_files:
        final_ds = xr.Dataset()
        ds = xr.open_dataset(utils.ensure_trailing_slash(extract_dir) + file)

        if file.endswith("MonthlyData.nc"):
            # insitu data
            final_ds["std_dev"] = ds.value_std_dev
            final_ds["numb"] = ds.nvalue
            final_ds["value_var"] = ds.value.values
        elif file.endswith("event.nc"):
            # flask data
            final_ds = stats_from_events(ds)
        else:
            # skip all other files in zip-folder
            continue

        final_ds["year"] = ds.time.dt.year.values
        final_ds["month"] = ds.time.dt.month.values
        final_ds["latitude"] = ds.latitude.values
        final_ds["longitude"] = ds.longitude.values
        final_ds["site_code"] = ds.attrs["site_code"]
        final_ds["network"] = "noaa"
        final_ds["altitude"] = ds.altitude.values
        final_ds["insitu_vs_flask"] = ds.attrs["dataset_project"].split("-")[-1]
        final_ds["sampling_strategy"] = ds.attrs["dataset_project"]
        final_ds["gas"] = ds.attrs["dataset_parameter"]
        final_ds["unit"] = np.where(
            ds.attrs["dataset_parameter"] == "ch4", "ppb", "ppm"
        )
        final_ds["version"] = ds.attrs["dataset_creation_date"]
        final_ds["instrument"] = "noaa"

        # fill values to NAN
        final_ds["value"] = final_ds["value"].where(final_ds["value"] >= 0, np.nan)

        final_ds = final_ds.assign_coords(
            {
                "year": np.unique(final_ds.year),
                "month": np.unique(final_ds.month),
                "latitude": np.unique(final_ds.latitude),
                "longitude": np.unique(final_ds.longitude),
                "altitude": np.unique(final_ds.altitude),
            }
        )

        df_list.append(final_ds)

    ds_combined = xr.concat(df_list, dim="source", join="outer")

    return ds_combined


@flow(name="download_surface_data", description="Download and preprocess surface data")
def download_surface_data(gas: str, save_to_path: str = "data/downloads"):
    """
    Download and preprocess surface GHG concentration

    Parameters
    ----------
    gas :
        greenhouse gas,
        either ch4 or co2

    save_to_path :
        path to save the results
    """
    ds_all = []

    for sampling in ["insitu", "flask"]:
        download_zip_from_noaa(
            gas=gas, sampling_strategy=sampling, save_to_path=save_to_path
        )

        unzip_download(
            zip_path=f"data/downloads/noaa_{gas}_surface_{sampling}.zip",
            extract_dir=f"data/downloads/{gas}/temp",
        )

        ds_all.append(
            merge_netCDFs(
                extract_dir=f"data/downloads/{gas}/temp/{gas}_surface-{sampling}_ccgg_netCDF"
            )
        )

    # ds_raw = xr.concat(ds_all, dim="sampling", join="outer")
    ## TODO: combine:netcdfs for insitu data


if __name__ == "__main__":
    download_surface_data("ch4")
