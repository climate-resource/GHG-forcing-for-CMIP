"""
Download ground-based data

The task of this module is data scraping of the GHG
concentration data from (A)GAGE and NOAA networks
"""

import logging
import os
from pathlib import Path
from typing import Any, Union

import httpx
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
import xarray as xr
from prefect import flow, task

from ghg_forcing_for_cmip import CONFIG, utils, validation

logging.basicConfig(
    level=logging.INFO,  # Default level
    format="%(levelname)s: %(message)s",
)


@task(description="Download zip-folder from NOAA", cache_policy=CONFIG.CACHE_POLICIES)
def download_zip_from_noaa(
    gas: str, sampling_strategy: str, save_to_path: Path
) -> None:
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
    os.makedirs(save_to_path, exist_ok=True)

    if sampling_strategy == "insitu":
        folder, sampling_strategy = "in-situ", "insitu"
    if sampling_strategy == "flask":
        folder, sampling_strategy = "flask", "flask"

    url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{folder}/surface/{gas}_surface-{sampling_strategy}_ccgg_netCDF.zip"

    # note: probably timeout has to be adjusted (currently only an initial guess)
    response = requests.get(url, timeout=10)

    with open(save_to_path / f"noaa_{gas}_surface_{sampling_strategy}.zip", "wb") as f:
        f.write(response.content)

    logging.info(f"downloaded NOAA-zip ({gas}-{sampling_strategy}) to {save_to_path!s}")


def stats_from_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics from event-data

    Compute mean, std_dev, and count values across
    events for insitu data.

    Parameters
    ----------
    df :
        insitu data

    Returns
    -------
    :
        data including mean value, std, and count
        across events for insitu data
    """
    df["year"] = df.time.dt.year.values
    df["month"] = df.time.dt.month.values

    df = (
        df.groupby(["year", "month", "latitude", "longitude", "altitude"])
        .agg({"value": ["mean", "std", "count"]})
        .reset_index()
    )

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

    return df


@task(
    description="Download methane data from AGAGE network",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def download_agage(save_to_path: Path) -> None:
    """
    Download methane concentrations from (A)GAGE database

    Parameters
    ----------
    save_to_path:
        path where data should be stored

    """
    os.makedirs(save_to_path, exist_ok=True)

    r_compounds = httpx.get(
        "https://www-air.larc.nasa.gov/missions/agage/api/data/compounds"
    )
    # check response
    r_compounds.raise_for_status()
    # get id for methane
    r_compunds_dict = pd.DataFrame(r_compounds.json())
    compound_id = r_compunds_dict[r_compunds_dict.compound_name == "Methane"][
        "id"
    ].values[0]

    # Get files available for extracted id
    r_files = []
    page_number = 1
    while True:
        try:
            httpx.get(
                f"https://www-air.larc.nasa.gov/missions/agage/api/data/{page_number}",
                params={
                    "recommended": True,
                    "compound": compound_id,
                    "data_frequency": 2,
                    "product_type": 1,
                },
            ).raise_for_status()

        except httpx.HTTPStatusError:
            break
        else:
            # data_frequency: 2 stands for "monthly"
            # product_type: 1 stands for "mole fraction"
            r_file = httpx.get(
                f"https://www-air.larc.nasa.gov/missions/agage/api/data/{page_number}",
                params={
                    "recommended": True,
                    "compound": compound_id,
                    "data_frequency": 2,
                    "product_type": 1,
                },
            )
            page_number += 1

        r_files.append(pd.DataFrame(r_file.json()))

    file_ids = pd.concat(r_files)

    # check that all files are included
    if len(file_ids) != file_ids["count"].unique()[0]:
        raise ValueError(  # noqa: TRY003
            "length of extracted data files does not correspond to database-counts"
        )

    # download netCDF zip.files
    for file_id, file_name in zip(file_ids.id, file_ids.file_name):
        response = requests.get(
            f"https://www-air.larc.nasa.gov/missions/agage/api/data/download/{file_id}",
            timeout=10,
        )

        with open(save_to_path / file_name.replace(".nc", ".zip"), "wb") as f:
            f.write(response.content)

    logging.info(
        f"downloaded AGAGE-zip to {save_to_path / file_name.replace('.nc', '.zip')!s}"
    )


@task(
    description="unzip and postprocess AGAGE data",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def postprocess_agage(zip_path: Path, extract_dir: Path) -> pd.DataFrame:
    """
    Unzip and merge single AGAGE data files

    Parameters
    ----------
    zip_path :
        Path to the zip file (e.g., "data/downloads/")

    extract_dir :
        path to extracted netCDF files

    Returns
    -------
    :
        combined data files from single AGAGE datasets
    """
    # unzip
    files = os.listdir(zip_path)
    for file in files:
        if file.endswith(".zip") and file.startswith("agage"):
            utils.unzip_download(zip_path / file, extract_dir)

    df_agage = merge_netCDFs(extract_dir)

    return df_agage.dropna(subset="value")


@task(
    description="Merge information from single files into one single netCDF",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def merge_netCDFs(
    extract_dir: Path,
) -> pd.DataFrame:
    """
    Combine netCDF files into a single dataframe

    Parameters
    ----------
    extract_dir :
        path to extracted netCDF files

    Returns
    -------
    :
        combined dataframe
    """
    files = os.listdir(extract_dir)
    nc_files = [file for file in files if file.endswith(".nc")]

    df_list = []
    for file in nc_files:
        if (
            file.endswith("MonthlyData.nc")
            or file.endswith("event.nc")
            or file.startswith("agage")
        ):
            final_df = pd.DataFrame()
            ds = xr.open_dataset(extract_dir / file)
            df_raw = ds.to_dataframe().reset_index()

            if not file.startswith("agage"):
                # maintain only values with valid quality flag
                df = df_raw[df_raw.qcflag == bytes("...", encoding="utf")].reset_index(
                    drop=True
                )
            else:
                df = df_raw

            if file.startswith("agage"):
                network = "agage"

                final_df["std_dev"] = df.mf_variability.values
                final_df["numb"] = df.mf_count.values
                final_df["value"] = df.mf.values
                final_df["year"] = df.time.dt.year.values
                final_df["month"] = df.time.dt.month.values
                final_df["latitude"] = ds.attrs["inlet_latitude"]
                final_df["longitude"] = ds.attrs["inlet_longitude"]
                final_df["altitude"] = df.inlet_height.values

            if file.endswith("MonthlyData.nc"):
                network = "noaa"
                # insitu data
                final_df["std_dev"] = df.value_std_dev.values
                final_df["numb"] = df.nvalue.values
                final_df["value"] = df.value.values
                final_df["year"] = df.time.dt.year.values
                final_df["month"] = df.time.dt.month.values
                final_df["latitude"] = df.latitude.values
                final_df["longitude"] = df.longitude.values
                final_df["altitude"] = df.altitude.values

            if file.endswith("event.nc"):
                network = "noaa"
                # flask data
                final_df = stats_from_events(df)

            final_df["site_code"] = ds.attrs["site_code"]
            final_df["network"] = network
            final_df["insitu_vs_flask"] = (
                ds.attrs["dataset_project"].split("-")[-1]
                if network == "noaa"
                else np.nan
            )
            final_df["sampling_strategy"] = (
                file.split("_")[2] if network == "noaa" else np.nan
            )
            final_df["gas"] = (
                ds.attrs["dataset_parameter"] if network == "noaa" else "ch4"
            )
            final_df["unit"] = "ppb" if final_df["gas"].unique() == "ch4" else "ppm"
            final_df["version"] = (
                ds.attrs["dataset_creation_date"]
                if network == "noaa"
                else ds.attrs["version"]
            )
            final_df["instrument"] = (
                "noaa" if network == "noaa" else ds.attrs["instrument"]
            )
            final_df["value"] = np.where(
                final_df["value"] < 0.0, np.nan, final_df["value"]
            )

            df_list.append(final_df)

    df_combined = pd.concat(df_list)

    return df_combined


def compute_bounds(
    values_idx: npt.NDArray[Union[np.int_, np.float32]],
    bounds: npt.NDArray[Union[np.int_, np.float32]],
    boundary_val: Union[int, float],
) -> tuple[
    npt.NDArray[Union[np.int_, np.float32]], npt.NDArray[Union[np.int_, np.float32]]
]:
    """
    Compute lower and upper boundary of grid cell

    helper function for add_lat_lon_bnds()

    Parameters
    ----------
    values_idx :
        np.array of indices into bounds

    bounds     :
        np.array of bin edges

    boundary_val :
        boundary values for min/max

    Returns
    -------
    :
        lower boundary, upper boundary
    """
    # lower bound
    if values_idx == np.where(bounds == -boundary_val)[0]:
        lower = bounds[values_idx + 1]
    else:
        lower = np.where(
            bounds[values_idx] < 0, bounds[values_idx], bounds[values_idx - 1]
        )

    # upper
    if np.abs(bounds[values_idx]) == boundary_val:
        upper = bounds[values_idx]
    else:
        upper = np.where(
            bounds[values_idx] < 0, bounds[values_idx - 1], bounds[values_idx]
        )

    return lower, upper


def get_indices(
    values: "pd.Series[Any]", bounds: npt.NDArray[Union[np.int_, np.float32]]
) -> npt.NDArray[Union[np.int_, np.float32]]:
    """
    Compute indices based on CONFIG.LAT/LON_BINS

    helper function for add_lat_lon_bnds()

    Parameters
    ----------
    values :
        latitude/longitude series from data frame

    bounds :
        latitude/longitude bins as set in CONFIG file

    Returns
    -------
    :
        array with indices
    """
    indices = []
    for val in values:
        if val in bounds:
            indices.append(np.where(bounds == val)[0][0])
        else:
            indices.append(np.searchsorted(bounds, val, side="right"))
    return np.array(indices)


@task(
    name="add_lat_lon_bnds",
    description="Add latitude and longitude bands",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def add_lat_lon_bnds(d_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Add latitude and longitude boundaries to d_combined

    Parameters
    ----------
    d_combined :
        combined dataframe

    Returns
    -------
    :
        combined dataframe with latitude and longitude boundaries
    """
    get_lat_idx = get_indices(d_combined.latitude, CONFIG.LAT_BIN_BOUNDS)
    get_lon_idx = get_indices(d_combined.longitude, CONFIG.LON_BIN_BOUNDS)

    # compute lon bounds
    # TODO: vectorization in for-loop not optimal; don't have a better solution yet
    d_combined["lon_bnd/lower"], d_combined["lon_bnd/upper"] = np.stack(
        [compute_bounds(idx, CONFIG.LON_BIN_BOUNDS, 180) for idx in get_lon_idx], -1
    )

    # compute lat bounds
    d_combined["lat_bnd/lower"], d_combined["lat_bnd/upper"] = np.stack(
        [compute_bounds(idx, CONFIG.LAT_BIN_BOUNDS, 90) for idx in get_lat_idx], -1
    )

    d_combined["lat"] = (d_combined["lat_bnd/lower"] + d_combined["lat_bnd/upper"]) / 2
    d_combined["lon"] = (d_combined["lon_bnd/lower"] + d_combined["lon_bnd/upper"]) / 2

    d_bnd = d_combined.melt(
        id_vars=[col for col in d_combined.columns if "bnd" not in col],
        var_name="coord_bnd",
        value_name="bnd_value",
    )
    d_bnd["coord"] = d_bnd["coord_bnd"].apply(lambda x: x.split("/")[0])
    d_bnd["bnd"] = d_bnd["coord_bnd"].apply(lambda x: 0 if "lower" in x else 1)

    d_bnd.drop(columns="coord_bnd", inplace=True)
    d_bnd.drop_duplicates(inplace=True)

    d_reshaped = d_bnd.pivot(
        index=[col for col in d_bnd.columns if col not in ["bnd_value", "coord"]],
        columns="coord",
        values="bnd_value",
    ).reset_index()

    return d_reshaped


@task(
    name="validate_surface_data",
    description="validate surface data",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def validate_surface_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate column types of dataframe using type schema

    Parameters
    ----------
    df :
        postprocessed dataframe

    Returns
    -------
    :
        final validated dataframe
    """
    df["time_fractional"] = df.year + df.month / 12
    df["time"] = pd.to_datetime(
        pd.DataFrame({"year": df.year, "month": df.month, "day": 15, "hour": 12}),
        utc=True,
    )
    df["year"] = df.year.astype(np.int64)
    df["month"] = df.month.astype(np.int64)
    df["latitude"] = df.latitude.astype(np.float64)
    df["longitude"] = df.longitude.astype(np.float64)
    df["lat_bnd"] = df.lat_bnd.astype(np.int64)
    df["lon_bnd"] = df.lon_bnd.astype(np.int64)
    df["lat"] = df.lat.astype(np.float64)
    df["lon"] = df.lon.astype(np.float64)
    df["value"] = df.value.astype(np.float64)
    df["std_dev"] = df.std_dev.astype(np.float64)
    df["numb"] = df.numb.astype(np.int64)
    df["site_code"] = df.site_code.astype(str)
    df["network"] = df.network.astype(str)
    df["altitude"] = df.altitude.astype(np.float64)
    df["gas"] = df.gas.astype(str)
    df["unit"] = df.unit.astype(str)
    df["version"] = df.version.astype(str)
    df["instrument"] = df.instrument.astype(str)

    validation.validate_gb_dataframe(df)

    return df


@flow(name="download_surface_data", description="Download and preprocess surface data")
def download_surface_data(
    gas: str, remove_original_files: bool, save_to_path: str = "data/downloads"
) -> None:
    """
    Download and preprocess surface GHG concentration

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

    df_all = []
    # AGAGE network
    download_agage(save_to_path=save_to_path_arg / "ch4/original/agage")

    df_agage = postprocess_agage(
        zip_path=save_to_path_arg / "ch4/original/agage",
        extract_dir=save_to_path_arg / "ch4/original/agage",
    )

    # NOAA network
    for sampling in ["flask", "insitu"]:
        download_zip_from_noaa.with_options(name=f"download_noaa_zip_{gas}_{sampling}")(
            gas=gas, sampling_strategy=sampling, save_to_path=save_to_path_arg
        )

        utils.unzip_download.with_options(name=f"unzip_download_{gas}_{sampling}")(
            zip_path=save_to_path_arg / f"noaa_{gas}_surface_{sampling}.zip",
            extract_dir=save_to_path_arg / f"{gas}/original",
        )

        df_all.append(
            merge_netCDFs.with_options(name=f"merge_netCDFs_{gas}_{sampling}")(
                extract_dir=save_to_path_arg
                / f"{gas}/original/{gas}_surface-{sampling}_ccgg_netCDF"
            )
        )

    # agage only for CH4 not for CO2
    if gas == "ch4":
        df_combined = pd.concat([*df_all, df_agage])
    else:
        df_combined = pd.concat([*df_all])

    # add bins for latitudes, longitudes
    df_processed = add_lat_lon_bnds(df_combined)

    # postprocess and validate created dataframe
    df_final = validate_surface_data(df_processed)

    # clean up repo and save file
    utils.clean_and_save(
        df_final,
        gas=gas,
        save_to_path=save_to_path_arg,
        measurement_type="gb",
        remove_original_files=remove_original_files,
    )


if __name__ == "__main__":
    download_surface_data(gas="co2", remove_original_files=False)
