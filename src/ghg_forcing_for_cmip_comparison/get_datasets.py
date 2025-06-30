"""
Download obs4mips and cmip data sets
"""

import glob
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import cdsapi  # type: ignore
import numpy as np
import pandas as pd
import requests
from prefect import flow, task
from prefect.cache_policies import INPUTS, TASK_SOURCE

from ghg_forcing_for_cmip_comparison import utils

CACHE_POLICIES = TASK_SOURCE + INPUTS


@task(
    name="download_zip_from_cds",
    description="Download obs4mips zip from Climate Data Store",
    task_run_name="download_zip_from_cds_{gas}",
    cache_policy=CACHE_POLICIES,
    refresh_cache=True,
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
    cache_policy=CACHE_POLICIES,
)
def unzip_download(pattern: str, path_to_zip: str, path_to_file: str) -> None:
    """
    Unzip downloaded data folder

    Parameters
    ----------
    pattern :
        unique identifier that discriminates
        datasets from each other

    path_to_zip :
        path to downloaded data

    path_to_file :
        path to unzipped data file

    """
    files = os.listdir(path_to_zip)
    match_string = re.compile(f"{pattern}")

    # Find the first matching file or raise an error if none found
    target_file = next((f for f in files if match_string.match(f)), None)
    if target_file is None:
        raise FileNotFoundError(  # noqa: TRY003
            f"No file matching pattern {pattern} found in {path_to_zip}"
        )

    zip_path = os.path.join(path_to_zip, target_file)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path_to_file)

    return print(f"unzipped data {pattern} to {path_to_file}")


@task(
    name="download_zip_from_noaa_archive",
    description="Download zip from NOAA archive link",
    cache_policy=CACHE_POLICIES,
)
def download_noaa(gas: str, save_to_path: str = "data/downloads") -> None:
    """
    Download NOAA data

    Parameters
    ----------
    gas :
        target greenhouse gas variable; either 'co2' or 'ch4'

    save_to_path :
        path to save downloaded data
    """
    for dir, sampling_strategy in zip(["in-situ", "flask"], ["insitu", "flask"]):
        url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{dir}/surface/{gas}_surface-{sampling_strategy}_ccgg_text.zip"
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
    cache_policy=CACHE_POLICIES,
)
def download_agage(
    save_to_path: str = "data/downloads", save_file_suffix: str = ""
) -> None:
    """
    Download agage and gage text files from online data archive

    Parameters
    ----------
    save_to_path :
        path to save downloaded data

    save_file_suffix :
        suffix to add to file name
    """
    AGAGE_URLS = [
        "https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly_mean/barbados/ascii/AGAGE-GCMD_RPB_ch4_mon.txt",
        "https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly_mean/capegrim/ascii/AGAGE-GCMD_CGO_ch4_mon.txt",
        "https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly_mean/macehead/ascii/AGAGE-GCMD_MHD_ch4_mon.txt",
        "https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly_mean/samoa/ascii/AGAGE-GCMD_SMO_ch4_mon.txt",
        "https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly_mean/trinidad/ascii/AGAGE-GCMD_THD_ch4_mon.txt",
    ]

    GAGE_URLS = [
        "https://agage2.eas.gatech.edu/data_archive/gage/monthly/CGO-gage.mon",
        "https://agage2.eas.gatech.edu/data_archive/gage/monthly/MHD-gage.mon",
        "https://agage2.eas.gatech.edu/data_archive/gage/monthly/ORG-gage.mon",
        "https://agage2.eas.gatech.edu/data_archive/gage/monthly/RPB-gage.mon",
        "https://agage2.eas.gatech.edu/data_archive/gage/monthly/SMO-gage.mon",
    ]

    for network in ["agage", "gage"]:
        if network == "agage":
            URLS = AGAGE_URLS
        if network == "gage":
            URLS = GAGE_URLS

        save_to_path2 = f"{save_to_path}/ch4/{network}"
        # create directory if it doesn't exist
        os.makedirs(save_to_path2, exist_ok=True)

        for url in URLS:
            file_name = url.split("/")[-1].removesuffix(".txt")
            response = requests.get(url)  # noqa: S113

            if response.status_code == 200:  # noqa: PLR2004
                with open(
                    save_to_path2 + "/" + file_name + save_file_suffix, "wb"
                ) as f:
                    f.write(response.content)
                print(f"downloaded {network.upper()} {file_name} to {save_to_path2}")
            else:
                print(f"Failed to download. Status code: {response.status_code}")


@task(
    name="save_txt_as_df_csv",
    description="Save txt file as pandas dataframe csv",
    cache_policy=CACHE_POLICIES,
)
def txt_to_csv_file(
    file_path: str,
    save_to_path: str = "data/downloads",
    columns_in_comment: bool = False,
    hardcode_skip_rows: Optional[int] = None,
) -> None:
    """
    Read text file and convert to pandas dataframes

    Parameters
    ----------
    file_path :
        path to text file

    save_to_path :
        path to save downloaded data

    columns_in_comment :
        whether column names are included in comments starting with '#'

    hardcode_skip_rows :
        hardcode number of rows to skip in text file
    """
    file_name = file_path.split("/")[-1].removesuffix(".txt")

    # create directory if it doesn't exist
    create_sub_path = save_to_path + "/csv/"
    os.makedirs(create_sub_path, exist_ok=True)

    # Find the first non-comment line (column headers)
    with open(file_path) as file:
        for i, line in enumerate(file):
            if not line.startswith("#"):
                header_line = i
                break

    if columns_in_comment:
        header_line = header_line - 1

    if hardcode_skip_rows is not None:
        header_line = hardcode_skip_rows

    df = pd.read_csv(file_path, sep=r"\s+", skiprows=header_line, engine="python")

    # If the comment symbol '#' is mistakenly read as a column name, clean it
    if columns_in_comment:
        df.columns = [col for col in df.columns if col not in ["#", "dev"]] + [""]
        df = df.iloc[:, :-2]  # remove two last (empty) columns
        df.rename(columns={"dev.": "numb", "std.": "std_dev"}, inplace=True)
        df["site_code"] = file_name.split("_")[1]
        df["network"] = "agage"
        df["sampling_strategy"] = pd.NA

    if hardcode_skip_rows is not None:
        df["site_code"] = file_name.split("-")[0]
        df["network"] = "gage"
        df["sampling_strategy"] = pd.NA

    if (not columns_in_comment) and (hardcode_skip_rows is None):
        df["network"] = "noaa"
        df["sampling_strategy"] = file_name.split("_")[2]

    df.to_csv(create_sub_path + file_name)


@task(
    name="loop_over_txt_files",
    description="Loop over txt files in stored folder",
    task_run_name="loop_over_txt_files_in_folder-{folder_pattern}",
    cache_policy=CACHE_POLICIES,
)
def txt_to_csv_folder(
    folder_pattern: str,
    file_endings: tuple[str, ...],
    path_to_dir: str = "data/downloads",
    columns_in_comment: bool = False,
    hardcode_skip_rows: Optional[int] = None,
) -> None:
    """
    Extract relevant text files and convert to pandas dataframes

    Parameters
    ----------
    folder_pattern :
        unique identifier that discriminates relevant folders

    file_endings :
        unique identifier that discriminates relevant files

    path_to_dir :
        path to downloaded data

    columns_in_comment :
        whether column names are included in comments starting with '#'

    hardcode_skip_rows :
        hardcode number of rows to skip in text file
    """
    files = os.listdir(path_to_dir)

    match_string = re.compile(f"{folder_pattern}")

    target_folder = next((f for f in files if match_string.match(f)), None)

    save_subdir = path_to_dir + "/" + target_folder
    os.makedirs(save_subdir, exist_ok=True)

    for file in os.listdir(path_to_dir + "/" + target_folder):
        if file.endswith(file_endings):
            txt_to_csv_file.with_options(task_run_name=f"save_txt_as_df_csv-{file}")(
                file_path=path_to_dir + "/" + target_folder + "/" + file,
                save_to_path=save_subdir,
                columns_in_comment=columns_in_comment,
                hardcode_skip_rows=hardcode_skip_rows,
            )


@task(
    name="combine_csv_files_to_clean",
    description="Combine csv files to final csv",
    task_run_name="combine_csv_files_to_final-{gas}",
    cache_policy=CACHE_POLICIES,
)
def combine_csv_files(
    gas: str,
    save_file_suffix: str,
    path_to_csv: str = "data/downloads/csv",
    path_to_save: str = "data/downloads",
) -> None:
    """
    Combine CSV files

    Parameters
    ----------
    gas :
        target greenhouse gas variable

    save_file_suffix :
        suffix provided to downloaded data file

    path_to_csv :
        path where single csv files are saved

    path_to_save :
        path to save combined csv file
    """
    csv_folder = os.listdir(path_to_csv)

    os.makedirs(path_to_save, exist_ok=True)

    all_dfs = []
    for file in csv_folder:
        if gas in file:
            df = pd.read_csv(path_to_csv + "/" + file)
            if "insitu" in file:
                df.rename(columns={"value_std_dev": "value_unc"}, inplace=True)
            all_dfs.append(df)

    pd.concat(all_dfs).to_csv(path_to_save + f"/{gas}_{save_file_suffix}.csv")


@task(
    name="compute_total_monthly_std",
    description="Compute total monthly std. as sum of value std and instrument std.",
    cache_policy=CACHE_POLICIES,
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
    d_grouped.columns = [
        "_".join(col).strip() if len(col[1]) != 0 else col[0]
        for col in d_grouped.columns
    ]
    # compute total variance as sum of instrument-variance and value-variance
    d_grouped["total_var"] = d_grouped["value_var"] + d_grouped["instrument_var_mean"]
    # derive total standard deviation
    d_grouped["std_dev"] = np.sqrt(d_grouped["total_var"])
    # if data is already monthly data, use count-value and std-value
    # as provided in dataset
    d_grouped["numb"] = np.where(
        d_grouped.value_count == 1, d_grouped.nvalue_mean, d_grouped.value_count
    )
    d_grouped["std_dev"] = np.where(
        d_grouped.std_dev.isna(),
        np.sqrt(d_grouped.instrument_var_mean),
        d_grouped.std_dev,
    )
    # maintain relevant variables only
    d_grouped.drop(
        columns=[
            "instrument_var_mean",
            "total_var",
            "value_var",
            "value_count",
            "nvalue_mean",
        ],
        inplace=True,
    )
    d_grouped.rename(columns={"value_mean": "value"}, inplace=True)

    d_grouped.to_csv(path_to_csv.replace("_noaa.csv", "_noaa_processed.csv"))


@task(
    name="combine_final_csv",
    description="Combine final csv files",
    cache_policy=CACHE_POLICIES,
)
def combine_final_csv(gas: str, path_to_save: str = "data/downloads") -> None:
    """
    Combine final csv files from NOAA, AGAGE, and GAGE networks

    Parameters
    ----------
    gas :
        target greenhouse gas variable

    path_to_save :
        path to save combined csv files
    """
    lat = {
        "CGO": -40.6833,
        "MHD": 53.3266,
        "RPB": 13.1651,
        "SMO": -14.2474,
        "THD": 41.0541,
        "ORG": 45.0000,
    }
    lon = {
        "CGO": 144.6894,
        "MHD": -9.9045,
        "RPB": -59.4320,
        "SMO": -170.5644,
        "THD": -124.1510,
        "ORG": 124.0000,
    }

    if gas == "ch4":
        df1 = pd.read_csv(path_to_save + f"/{gas}/agage/clean/{gas}_agage.csv")
        df1.dropna(subset="mean", inplace=True)
        df1 = df1[[col for col in df1.columns if not col.startswith("Unnamed")]]
        df1.rename(columns={"mean": "value", "std": "std_dev"}, inplace=True)
        df1["latitude"] = df1["site_code"].map(lat)
        df1["longitude"] = df1["site_code"].map(lon)

        df2 = pd.read_csv(path_to_save + f"/{gas}/gage/clean/{gas}_gage.csv")
        df2 = df2[
            ["time", "MM", "YYYY", "numb.7", "CH4", "std..7", "site_code", "network"]
        ]
        df2.rename(
            columns={
                "numb.7": "numb",
                "std..7": "std_dev",
                "MM": "month",
                "YYYY": "year",
                "CH4": "value",
            },
            inplace=True,
        )
        df2 = df2[(df2.value != 0.0)]

        df2["latitude"] = df2["site_code"].map(lat)
        df2["longitude"] = df2["site_code"].map(lon)
        df_combined = pd.concat([df1, df2])

    dfs_noaa = []
    for sampling_type in ["flask", "insitu"]:
        df3 = pd.read_csv(
            path_to_save
            + f"/{gas}/noaa/{gas}_surface-{sampling_type}_ccgg_text/"
            + f"clean/{gas}_noaa_processed.csv"
        )
        df3["insitu_vs_flask"] = sampling_type
        dfs_noaa.append(df3)

    df_noaa_combined = pd.concat(dfs_noaa)
    df_noaa_combined.drop(columns=["Unnamed: 0"], inplace=True)
    df_noaa_combined["time"] = np.round(
        df_noaa_combined["year"] + (df_noaa_combined["month"] - 0.5) / 12, decimals=3
    )
    if gas == "ch4":
        d_combined = pd.concat([df_combined, df_noaa_combined])
    else:
        d_combined = df_noaa_combined

    # this condition is applied due to the raw data from MLO
    # surface-insitu dataset in 2022/2023 are a few measures
    # that have a non-zero value, but a std and nvalue of zero.
    # as this makes no sense I delete these measurements entirely (row)
    d_combined[d_combined.numb != 0.0].to_csv(path_to_save + f"/{gas}/{gas}_raw.csv")


@task(
    name="postprocess_cmip_data",
    description="postprocess cmip data and prepare for analysis",
    cache_policy=CACHE_POLICIES,
)
def postprocess_cmip_data(path_to_save: str, gas: str) -> None:
    """
    Validate column types of dataframe using type schema

    Parameters
    ----------
    path_to_save :
        path to save combined csv files

    gas :
        target greenhouse gas variable
    """
    df = pd.read_csv(path_to_save + f"/{gas}/{gas}_raw.csv")
    df["time_fractional"] = df.time.astype(np.float64)
    df["time"] = pd.to_datetime(
        {"year": df.year, "month": df.month, "day": 16, "hour": 12}, utc=True
    )
    df["year"] = df.year.astype(np.int64)
    df["month"] = df.month.astype(np.int64)
    df["latitude"] = df.latitude.astype(np.float64)
    df["longitude"] = df.longitude.astype(np.float64)
    df["value"] = df.value.astype(np.float64)
    df["std_dev"] = df.std_dev.astype(np.float64)
    df["numb"] = df.numb.astype(np.int64)
    df["site_code"] = df.site_code.astype(str)
    df["network"] = df.network.astype(str)
    df["altitude"] = df.altitude.astype(np.float64)
    df["gas"] = gas
    df["unit"] = np.where(gas == "ch4", "ppb", "ppm")

    utils.GroundDataSchema.validate(df)

    df.to_csv(path_to_save + f"/{gas}/{gas}_raw.csv")


@flow(name="get_cmip_data", description="Download and extract CMIP data")
def download_cmip_flow(save_to_path: str = "data/downloads") -> None:
    """
    Download and extract CMIP data

    Parameters
    ----------
    save_to_path :
        path to save downloaded data
    """
    # download data of GAGE and AGAGE network
    download_agage(
        save_to_path=save_to_path,
        save_file_suffix="_ch4",
    )

    for network in ["gage", "agage"]:
        if network == "gage":
            columns_in_comment = False
            hardcode_skip_rows = 5
        else:
            columns_in_comment = True
            hardcode_skip_rows = None

        txt_to_csv_folder(
            folder_pattern=network,
            file_endings=("mon_ch4",),
            path_to_dir=save_to_path + "/ch4",
            columns_in_comment=columns_in_comment,
            hardcode_skip_rows=hardcode_skip_rows,
        )

        combine_csv_files(
            gas="ch4",
            save_file_suffix=network,
            path_to_csv=save_to_path + f"/ch4/{network}/csv",
            path_to_save=save_to_path + f"/ch4/{network}/clean",
        )

    # download data of NOAA network
    for gas in ["co2", "ch4"]:
        download_noaa(gas=gas, save_to_path=save_to_path)

        for sampling_strategy in ["surface_insitu", "surface_flask"]:
            unzip_download(
                pattern=f"noaa_{gas}_{sampling_strategy}",
                path_to_zip=save_to_path,
                path_to_file=save_to_path + f"/{gas}/noaa/",
            )

            txt_to_csv_folder(
                folder_pattern=f"{gas}_{sampling_strategy.replace('_', '-')}",
                file_endings=("MonthlyData.txt", "event.txt"),
                path_to_dir=save_to_path + f"/{gas}/noaa/",
            )

            combine_csv_files(
                gas=gas,
                save_file_suffix="noaa",
                path_to_csv=save_to_path
                + f"/{gas}/noaa/{gas}_{sampling_strategy.replace('_', '-')}"
                + "_ccgg_text/csv",
                path_to_save=save_to_path
                + f"/{gas}/noaa/{gas}_{sampling_strategy.replace('_', '-')}"
                + "_ccgg_text/clean",
            )

            compute_total_monthly_std(
                path_to_csv=f"data/downloads/{gas}/noaa/{gas}_"
                + f"{sampling_strategy.replace('_', '-')}_ccgg_text"
                + f"/clean/{gas}_noaa.csv",
            )

        combine_final_csv(gas=gas, path_to_save=save_to_path)

        postprocess_cmip_data(path_to_save=save_to_path, gas=gas)


@flow(name="get_obs4mips_data", description="Download and extract OBS4MIPs data")
def download_obs4mips_flow(save_to_path: str = "data/downloads") -> None:
    """
    Download and extract OBS4MIPs data

    Parameters
    ----------
    save_to_path :
        path to save downloaded data
    """
    for gas in ("co2", "ch4"):
        make_api_request(gas=gas, save_to_path=save_to_path)

        unzip_download(
            pattern=f"obs4mips_x{gas}",
            path_to_zip=save_to_path,
            path_to_file=save_to_path + f"/{gas}",
        )


@flow(
    name="get_raw_datasets",
    description="Download and extract OBS4MIPs and ground-based datasets",
)
def get_data_flow(
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
    save_to_path :
        path to save downloaded data

    remove_zip_files :
        remove all zip files created during downloading and preprocessing

    remove_txt_files :
        remove all text files created during downloading and preprocessing
    """
    download_cmip_flow(save_to_path=save_to_path)

    download_obs4mips_flow(save_to_path=save_to_path)

    # clean-up downloading directory
    if remove_zip_files:
        for zip_file in glob.glob(os.path.join(save_to_path, "*.zip")):
            os.remove(zip_file)

    if remove_txt_files:
        for txt_file in Path(save_to_path).rglob("*.txt"):
            txt_file.unlink()
