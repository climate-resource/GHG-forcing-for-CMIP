import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """
    Delete downloaded test files and folders after tests passed
    """
    yield
    for test_folder in ["download_test", "download_test1", "download_test2"]:
        shutil.rmtree("tests/unit/" + test_folder, ignore_errors=True)
