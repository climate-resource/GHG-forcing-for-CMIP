"""
GHG forcing for CMIP

Compare CMIP7 data with earth observations as part of GHG forcing for CMIP project.
"""

import importlib.metadata

from ghg_forcing_for_cmip import (
    data_assimilation,
    data_comparison,
)

__version__ = importlib.metadata.version("ghg_forcing_for_cmip")

__all__ = [
    "data_assimilation",
    "data_comparison",
]
