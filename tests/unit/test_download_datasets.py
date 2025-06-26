import os

import pytest

from ghg_forcing_for_cmip_comparison.download_datasets import (
    make_api_request,
    unzip_obs4mips_download,
)


@pytest.mark.parametrize("input_gas", ("co2", "ch4"))
@pytest.mark.parametrize(
    "save_path", ("tests/unit/download_test1", "tests/unit/download_test2")
)
def test_make_api_request(input_gas, save_path, version="v4_5"):
    # download obs4mips data
    make_api_request(input_gas, save_path)

    # check whether zip file has been downloaded
    assert os.path.isfile(save_path + f"/obs4mips_x{input_gas}_{version}.zip")


# TODO: It is a bit unlucky that I have to hardcode the expected name of the file
@pytest.mark.parametrize(
    "input_gas, expected_file",
    [
        (
            "co2",
            "200301_202212-C3S-L3_XCO2-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.5.nc",
        ),
        (
            "ch4",
            "200301_202212-C3S-L3_XCH4-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.5.nc",
        ),
    ],
)
def test_unzip_obs4mips_download(
    input_gas, expected_file, save_path="tests/unit/download_test"
):
    # download obs4mips data
    make_api_request(input_gas, save_path)

    # unzip downloaded files
    unzip_obs4mips_download(
        input_gas, path_to_obs4mips_zip=save_path, path_to_obs4mips_nc=save_path
    )

    # check whether nc file has been extracted
    assert os.path.isfile(save_path + "/" + expected_file)
