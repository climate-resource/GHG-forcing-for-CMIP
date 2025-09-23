"""
Test download data

Unit tests for downloading GHG concentrations
from web APIs
"""

import os
import tempfile
import zipfile

import numpy as np
import pytest

from ghg_forcing_for_cmip.download_data import download_zip_from_noaa, unzip_download


@pytest.mark.parametrize(
    "gas,sampling_strategy",
    [
        ("co2", "in-situ"),
        ("ch4", "flask"),
    ],
)
def test_download_zip_from_noaa_real(gas, sampling_strategy, tmp_path):
    # Use temporary directory so no pollution
    save_dir = tmp_path

    # Run the actual function (downloads real NOAA data)
    download_zip_from_noaa(gas, sampling_strategy, save_to_path=str(save_dir))

    # Expected file name
    strat_name = "insitu" if sampling_strategy == "in-situ" else "flask"
    expected_file = save_dir / f"noaa_{gas}_surface_{strat_name}.zip"

    # Assertions
    np.testing.assert_(
        expected_file.exists(), f"Expected file {expected_file} was not created"
    )
    np.testing.assert_(expected_file.stat().st_size > 0, "Downloaded file is empty")


def test_unzip_download():
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Paths
        zip_path = os.path.join(temp_dir, "test.zip")
        extract_dir = os.path.join(temp_dir, "extracted")

        # Create a sample file to zip
        file_name = "hello.txt"
        file_content = b"Hello, World!"
        sample_file_path = os.path.join(temp_dir, file_name)
        with open(sample_file_path, "wb") as f:
            f.write(file_content)

        # Create a zip archive containing the sample file
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(sample_file_path, arcname=file_name)

        # Run the function under test
        unzip_download(zip_path, extract_dir)

        # Check that the file was extracted correctly
        extracted_file_path = os.path.join(extract_dir, file_name)

        np.testing.assert_(
            os.path.exists(extracted_file_path), "File was not extracted"
        )

        # Check that the original zip file was removed
        np.testing.assert_(not os.path.exists(zip_path), "Zip file was not deleted")
