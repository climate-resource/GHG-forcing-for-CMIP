"""
Global helper functions
"""

import os
from typing import Any, Optional

from prefect.tasks import task_input_hash


def is_pytest_running() -> bool:
    """
    Check whether Pytest is running
    """
    return "PYTEST_CURRENT_TEST" in os.environ


def custom_cache_key_fn() -> Optional[Any]:
    """
    Check whether results should be cached

    If Pytest is running, don't cache results
    otherwise use task_input_hash
    """
    if is_pytest_running():
        # Disable caching during pytest by returning None
        return None
    else:
        # Normal caching key, e.g. hash of inputs
        return task_input_hash
