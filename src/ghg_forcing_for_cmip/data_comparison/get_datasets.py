"""
Download obs4mips and cmip data sets
"""

import glob
import os
import shutil
import zipfile

import cdsapi  # type: ignore
import numpy as np
import pandas as pd
import requests
import xarray as xr
from prefect import flow, task

from ghg_forcing_for_cmip.data_comparison import CONFIG, utils


@task(
    name="download_zip_from_cds",
    description="Download obs4mips zip from Climate Data Store",
    task_run_name="download_zip_from_cds_{gas}",
    cache_policy=CONFIG.CACHE_POLICIES,
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
    name="unzip_download",
    description="Unzip downloaded data",
    refresh_cache=True,
    persist_result=False,
)
def unzip_download(path_to_zip: str, zip_folder: str, save_to: str) -> None:
    """
    Unzip downloaded data folder and delete zip file

    If files already exist, they will be deleted and re-downloaded

    Parameters
    ----------
    path_to_zip :
        path to downloaded data incl. zip file name

    zip_folder :
        path and name of downloaded zip folder

    save_to :
        path where to save unzipped data files

    """
    if os.path.exists(zip_folder):
        shutil.rmtree(zip_folder)

    with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
        zip_ref.extractall(save_to)

    os.remove(path_to_zip)
    return print(f"unzip {path_to_zip} to {save_to}")


@task(
    name="download_zip_from_noaa_archive",
    description="Download zip from NOAA archive link",
    refresh_cache=True,
    persist_result=False,
)
def download_noaa(
    gas: str,
    sampling_strategy: str,
    save_to_path: str = "data/downloads",
) -> None:
    """
    Download NOAA data as NETCDF zip-file

    Parameters
    ----------
    gas :
        target greenhouse gas variable; either 'co2' or 'ch4'

    sampling_strategy :
        either in-situ or flask

    save_to_path :
        path to save downloaded data
    """
    save_to_path = utils.ensure_trailing_slash(save_to_path)

    if sampling_strategy == "in-situ":
        dir, sampling_strategy = "in-situ", "insitu"
    if sampling_strategy == "flask":
        dir, sampling_strategy = "flask", "flask"

    url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{dir}/surface/{gas}_surface-{sampling_strategy}_ccgg_netCDF.zip"
    # create directory if it doesn't exist
    os.makedirs(save_to_path, exist_ok=True)

    response = requests.get(url)  # noqa: S113

    if response.status_code == 200:  # noqa: PLR2004
        with open(
            save_to_path + f"/noaa_{gas}_surface_{sampling_strategy}.zip", "wb"
        ) as f:
            f.write(response.content)
        print(f"downloaded NOAA {gas} {sampling_strategy} data to {save_to_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")


@task(
    name="download_txt_from_agage_archive",
    description="Download txt from AGAGE archive link",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def download_agage(
    path_to_source: str = "data/downloads/ch4/agage/agage_data",
) -> pd.DataFrame:
    """
    Download agage and gage data

    URL: https://www-air.larc.nasa.gov/missions/agage/data/
    data is based on the following selection criteria:
        - monthly baseline
        - methane
        - Version: 20250721
        - instruments: all
        - product type: mole fraction

    Parameters
    ----------
    path_to_source :
        path to source data folder (as downloaded from AGAGE archive)

    Returns
    -------
    :
        pandas DataFrame with Agage data
    """
    path_to_source = utils.ensure_trailing_slash(path_to_source)

    files = os.listdir(path_to_source)
    df_list = []
    for file in files:
        ds = xr.open_dataset(path_to_source + file)
        df = ds.to_dataframe().reset_index()
        df["year"] = df.time.dt.year
        df["month"] = df.time.dt.month
        df["latitude"] = float(ds.attrs["inlet_latitude"])
        df["longitude"] = float(ds.attrs["inlet_longitude"])
        df["site_code"] = str(ds.attrs["site_code"])
        df["network"] = str(ds.attrs["network"])
        df["altitude"] = float(ds.attrs["inlet_base_elevation_masl"])
        df["instrument"] = str(ds.attrs["instrument_type"])
        df["version"] = str(ds.attrs["version"])
        df["gas"] = str(ds.attrs["species"])
        df["sampling_strategy"] = None
        df["insitu_vs_flask"] = None
        df["unit"] = "ppb"

        df.rename(
            columns={"mf": "value", "mf_variability": "std_dev", "mf_count": "numb"},
            inplace=True,
        )

        df.drop(
            columns=[
                "mf_repeatability",
                "inlet_height",
                "sampling_period",
                "instrument_type",
            ],
            inplace=True,
        )

        df_list.append(df)

    return pd.concat(df_list)


@task(
    name="compute_total_monthly_std",
    description="Compute total monthly std. as sum of value std and instrument std.",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def compute_total_monthly_std(path_to_csv: str, fill_value: float = -999.99) -> None:
    """
    Compute total monthly std. dev.

    For flask noaa data this is the sum of value std and instrument std.
    For insitu noaa data we use the std provided in the data set
    (already monthly measurement)

    Parameters
    ----------
    path_to_csv :
        path to csv file with complete noaa data set

    fill_value :
        number used to indicate missing value
    """
    d = pd.read_csv(path_to_csv)
    # remove rows with missing values
    d = d[d.value != fill_value]
    # insert NAN for missing std. values
    d["value_unc"] = d.value_unc.where(d.value_unc >= 0.0, pd.NA)

    d.rename(columns={"value_unc": "instrument_std"}, inplace=True)
    d["instrument_var"] = d["instrument_std"] ** 2

    try:
        d["nvalue"]
    except KeyError:
        d["nvalue"] = 1

    # get monthly average
    d_grouped = (
        d[d.qcflag.str.startswith("..")]
        .groupby(
            [
                "site_code",
                "year",
                "month",
                "latitude",
                "longitude",
                "altitude",
                "network",
                "sampling_strategy",
            ]
        )
        .agg(
            {
                "value": ["mean", "var", "count"],
                "nvalue": "mean",
                "instrument_var": "mean",
            }
        )
        .reset_index()
    )

    # reset multi-index
    d_grouped.columns = pd.Index(
        [
            "_".join(col).strip() if len(col[1]) != 0 else col[0]
            for col in d_grouped.columns
        ]
    )
    # compute total variance as sum of instrument-variance and value-variance
    d_grouped["value_var"] = d_grouped["value_var"].fillna(0)
    d_grouped["total_var"] = d_grouped["value_var"] + d_grouped["instrument_var_mean"]
    # derive total standard deviation
    d_grouped["std_dev"] = np.sqrt(d_grouped.total_var.values.astype(np.float64))
    # if data have NaN std_dev use mean of year as imputation
    sd_mean_year = d_grouped.groupby("year").agg({"std_dev": "mean"}).reset_index()
    sd_mean_year.rename(columns={"std_dev": "std_dev_year"}, inplace=True)
    # add yearly averages to dataframe
    d_grouped = d_grouped.merge(sd_mean_year, on="year")
    d_grouped["std_dev"] = np.where(
        d_grouped.std_dev.isna(), d_grouped["std_dev_year"], d_grouped["std_dev"]
    )
    # if data is already monthly data, use count-value and std-value
    # as provided in dataset
    d_grouped["numb"] = np.where(
        d_grouped.value_count == 1, d_grouped.nvalue_mean, d_grouped.value_count
    )
    # maintain relevant variables only
    d_grouped.drop(
        columns=[
            "instrument_var_mean",
            "total_var",
            "value_var",
            "value_count",
            "nvalue_mean",
            "std_dev_year",
        ],
        inplace=True,
    )
    d_grouped.rename(columns={"value_mean": "value"}, inplace=True)

    d_grouped.to_csv(path_to_csv.replace("_noaa.csv", "_noaa_processed.csv"))


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
    get_lat_idx = np.searchsorted(
        CONFIG.LAT_BIN_BOUNDS, d_combined.latitude, side="right"
    )
    get_lon_idx = np.searchsorted(
        CONFIG.LON_BIN_BOUNDS, d_combined.longitude, side="right"
    )

    lat_neg = d_combined.latitude < 0
    lon_neg = d_combined.longitude < 0

    d_combined["lat_bnd/lower"] = CONFIG.LAT_BIN_BOUNDS[
        np.where(lat_neg, get_lat_idx, get_lat_idx - 1)
    ]
    d_combined["lat_bnd/upper"] = CONFIG.LAT_BIN_BOUNDS[
        np.where(lat_neg, get_lat_idx - 1, get_lat_idx)
    ]

    d_combined["lon_bnd/lower"] = CONFIG.LON_BIN_BOUNDS[
        np.where(lon_neg, get_lon_idx, get_lon_idx - 1)
    ]
    d_combined["lon_bnd/upper"] = CONFIG.LON_BIN_BOUNDS[
        np.where(lon_neg, get_lon_idx - 1, get_lon_idx)
    ]

    # this additional processing is done because of shipboard-flask measures
    # from noaa-POC which has exactly lon=-180.0 (max/min boundary) as coordinate
    d_combined.loc[d_combined.longitude == -180, "lon_bnd/upper"] = -180.0  # noqa: PLR2004
    d_combined.loc[d_combined.longitude == -180, "lon_bnd/lower"] = -175.0  # noqa: PLR2004
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

    d_reshaped = d_bnd.pivot(
        index=[col for col in d_bnd.columns if col not in ["bnd_value", "coord"]],
        columns="coord",
        values="bnd_value",
    ).reset_index()

    return d_reshaped


@task(
    name="postprocess_obs4mips_data",
    description="postprocess obs4mips data and prepare for analysis",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def postprocess_obs4mips_data(
    path_to_nc: str, gas: str = "co2", factor: float = 1e6
) -> None:
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
    all_files = os.listdir(path_to_nc + f"/{gas}/")
    ds = next(file for file in all_files if "OBS4MIPS" in file)

    df_raw = xr.open_dataset(path_to_nc + f"/{gas}/{ds}").to_dataframe().reset_index()
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

    utils.EODataSchema.validate(df)
    utils.save_data(df, path_to_nc, gas, "eo_raw")


@task(
    name="validate_cmip_data",
    description="validate cmip data",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def validate_cmip_data(df: pd.DataFrame) -> None:
    """
    Validate column types of dataframe using type schema

    Parameters
    ----------
    df :
        final pandas dataframe

    path_to_save :
        path to save file

    gas :
        target greenhouse gas variable
    """
    df["time_fractional"] = df.year + df.month / 12
    df["time"] = pd.to_datetime(
        pd.DataFrame({"year": df.year, "month": df.month, "day": 16, "hour": 12}),
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

    utils.GroundDataSchema.validate(df)


@task(name="combine_netCDFs", description="postprocess noaa data", refresh_cache=True)
def combine_netCDFs(
    path_to_files: str,
) -> pd.DataFrame:
    """
    Combine netCDF files into a single dataframe

    Parameters
    ----------
    path_to_files :
        path to extracted netCDF files from zip folder

    Returns
    -------
    :
        pandas dataframe
    """
    files = os.listdir(path_to_files)
    nc_files = [file for file in files if file.endswith(".nc")]

    df_list = []
    for file in nc_files:
        ds = xr.open_dataset(utils.ensure_trailing_slash(path_to_files) + file)
        df_full = ds.to_dataframe().reset_index()
        df = pd.DataFrame({})
        df["time"] = df_full.time
        df["year"] = df_full.time.dt.year
        df["month"] = df_full.time.dt.month
        df["latitude"] = df_full.latitude
        df["longitude"] = df_full.longitude
        df["value"] = df_full.value
        df["site_code"] = ds.attrs["site_code"]
        df["network"] = "noaa"
        df["altitude"] = df_full.altitude
        df["insitu_vs_flask"] = ds.attrs["dataset_project"].split("-")[-1]
        df["sampling_strategy"] = ds.attrs["dataset_project"]
        df["gas"] = ds.attrs["dataset_parameter"]
        df["unit"] = np.where(df.gas == "ch4", "ppb", "ppm")
        df["version"] = ds.attrs["dataset_creation_date"]
        df["instrument"] = "noaa"

        if file.endswith("MonthlyData.nc"):
            df["std_dev"] = df_full.value_std_dev
            df["numb"] = df_full.nvalue

        elif file.endswith("event.nc"):
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
            df["time"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

        else:
            continue

    # fill values to NAN
    df["value"] = df["value"].where(df.value >= 0, np.nan)

    df_list.append(df)
    df_combined = pd.concat(df_list)

    return df_combined.drop_duplicates()


@flow(name="get_cmip_data", description="Download and extract CMIP data")
def download_cmip_flow(gas: str, save_to_path: str = "data/downloads") -> None:
    """
    Download and extract CMIP data

    Parameters
    ----------
    gas :
        either ch4 or co2

    save_to_path :
        path to save downloaded data
    """
    save_to_path = utils.ensure_trailing_slash(save_to_path)

    if gas == "ch4":
        # download data of GAGE and AGAGE network
        df_agage = download_agage(
            path_to_source=save_to_path + "ch4/agage/agage_data",
        )
    else:
        df_agage = None

    # download data of NOAA network
    for sampling in ["insitu", "flask"]:
        download_noaa(
            gas=gas,
            sampling_strategy=str(np.where(sampling == "insitu", "in-situ", sampling)),
            save_to_path=save_to_path + gas + "/noaa",
        )
        unzip_download(
            path_to_zip=save_to_path + f"{gas}/noaa/noaa_{gas}_surface_{sampling}.zip",
            zip_folder=save_to_path
            + f"{gas}/noaa/{gas}_surface-{sampling}_ccgg_netCDF",
            save_to=save_to_path + gas + "/noaa",
        )

    # combine datasets
    df_insitu = combine_netCDFs(
        save_to_path + f"{gas}/noaa/{gas}_surface-insitu_ccgg_netCDF"
    )
    df_flask = combine_netCDFs(
        save_to_path + f"{gas}/noaa/{gas}_surface-flask_ccgg_netCDF"
    )

    df_combined = pd.concat([df_insitu, df_flask, df_agage])

    df_final = add_lat_lon_bnds(d_combined=df_combined)

    # validate columns and their types for the final dataframe
    validate_cmip_data(df_final)

    # save final dataset
    df_final.to_csv(save_to_path + gas + f"/{gas}_raw.csv", index=False)


@flow(name="get_obs4mips_data", description="Download and extract OBS4MIPs data")
def download_obs4mips_flow(gas: str, save_to_path: str = "data/downloads") -> None:
    """
    Download and extract OBS4MIPs data

    Parameters
    ----------
    gas :
        either co2 or ch4

    save_to_path :
        path to save downloaded data
    """
    make_api_request(gas=gas, save_to_path=save_to_path)

    unzip_download(
        pattern=f"obs4mips_x{gas}",
        path_to_zip=save_to_path,
        path_to_file=save_to_path + f"/{gas}",
    )

    postprocess_obs4mips_data(
        path_to=save_to_path, gas=gas, factor=np.where(gas == "ch4", 1e9, 1e6)
    )


@flow(
    name="get_raw_datasets",
    description="Download and extract OBS4MIPs and ground-based datasets",
)
def get_data_flow(
    gas: str,
    save_to_path: str = "data/downloads",
    remove_zip_files: bool = True,
    remove_txt_files: bool = True,
) -> None:
    """
    Run main flow for downloading and extracting data

    Downloading and extracting of OBS4MIPs and ground-based data
    for CH4 and CO2

    Parameters
    ----------
    gas :
        either ch4 or co2

    save_to_path :
        path to save downloaded data

    remove_zip_files :
        remove all zip files created during downloading and preprocessing

    remove_txt_files :
        remove all text files created during downloading and preprocessing
    """
    download_cmip_flow(gas=gas, save_to_path=save_to_path)

    download_obs4mips_flow(gas=gas, save_to_path=save_to_path)

    # clean-up downloading directory
    if remove_zip_files:
        for zip_file in glob.glob(os.path.join(save_to_path, "*.zip")):
            os.remove(zip_file)
