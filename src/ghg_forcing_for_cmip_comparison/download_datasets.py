"""
Download obs4mips and cmip data sets
"""

import os
import re
import zipfile
from typing import Optional

import cdsapi  # type: ignore
import pandas as pd
import requests
from prefect import flow, task


@task(
    name="download_obs4mips_data",
    description="Download obs4mips data from Climate Data Store",
    task_run_name="download_obs4mips_{gas}_data",
    cache_key_fn=None,
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
    name="unzip_downloaded_data",
    description="Unzip downloaded data",
    cache_key_fn=None,
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
    name="download_NOAA_data",
    description="Download NOAA data",
    cache_key_fn=None,
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
    for type in ["in-situ", "flask"]:
        if type == "in-situ":
            type1, type2 = "in-situ", "insitu"
        else:
            type1, type2 = "flask", "flask"

        url = f"https://gml.noaa.gov/aftp/data/greenhouse_gases/{gas}/{type1}/surface/{gas}_surface-{type2}_ccgg_text.zip"
        # create directory if it doesn't exist
        os.makedirs(save_to_path, exist_ok=True)

        response = requests.get(url)  # noqa: S113

        if response.status_code == 200:  # noqa: PLR2004
            with open(save_to_path + f"/noaa_{gas}_surface_{type2}.zip", "wb") as f:
                f.write(response.content)
            print(f"downloaded NOAA {gas} {type} data to {save_to_path}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")


@task(name="download_agege_txtfiles", description="Download (a)gege text files")
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
            path_to_file=save_to_path,
        )


@task(
    name="textfile_to_dataframes",
    description="Convert text files to pandas dataframes",
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
        df.columns = [col for col in df.columns if col != "#"] + [""]

    df.to_csv(create_sub_path + file_name)


@task(
    name="textfiles_to_dataframes",
    description="Convert text files to pandas dataframes",
    task_run_name="txt_to_df-{folder_pattern}",
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

    for file in os.listdir(path_to_dir + "/" + target_folder):
        if file.endswith(file_endings):
            txt_to_csv_file.with_options(task_run_name=f"text_to_csv-{file}")(
                file_path=path_to_dir + "/" + target_folder + "/" + file,
                save_to_path=path_to_dir + "/" + target_folder,
                columns_in_comment=columns_in_comment,
                hardcode_skip_rows=hardcode_skip_rows,
            )


@task(
    name="combine_csv_files",
    description="Combine CSV files",
    task_run_name="combined_csv_files-{gas}",
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

    all_dfs = []
    for file in csv_folder:
        if gas in file:
            df = pd.read_csv(path_to_csv + "/" + file)
            if "insitu" in file:
                df.rename(columns={"value_std_dev": "value_unc"}, inplace=True)
            all_dfs.append(df)

    pd.concat(all_dfs).to_csv(path_to_save + f"/{gas}_{save_file_suffix}")


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
            path_to_save=save_to_path + f"/ch4/{network}",
        )

    # download data of NOAA network
    for gas in ["co2", "ch4"]:
        download_noaa(gas=gas, save_to_path=save_to_path)

        for gas_type in ["insitu", "flask"]:
            unzip_download(
                pattern=f"noaa_{gas}_surface_{gas_type}",
                path_to_zip=save_to_path,
                path_to_file=save_to_path + f"/{gas}/noaa/",
            )

            txt_to_csv_folder(
                folder_pattern=f"{gas}_surface-{gas_type}",
                file_endings=("MonthlyData.txt", "event.txt"),
                path_to_dir=save_to_path + f"/{gas}/noaa/",
            )

            combine_csv_files(
                gas=gas,
                save_file_suffix="noaa",
                path_to_csv=save_to_path
                + f"/{gas}/noaa/{gas}_surface-{gas_type}_ccgg_text/csv",
                path_to_save=save_to_path
                + f"/{gas}/noaa/{gas}_surface-{gas_type}_ccgg_text",
            )


if __name__ == "__main__":
    download_cmip_flow()
