"""
Test utils.py

Unit tests for helper functions
"""

import pytest

from ghg_forcing_for_cmip.utils import ensure_trailing_slash


@pytest.mark.parametrize(
    "test_path, expected",
    [
        ("test_path_without", "test_path_without/"),
        ("test_path_with/", "test_path_with/"),
        (".", "./"),
        ("/", "/"),
    ],
)
def test_ensure_trailing_slash(test_path, expected):
    observed = ensure_trailing_slash(test_path)

    assert observed == expected, (
        f"The observed test-path: {observed},"
        " does not match the expected path: {expected}"
    )
