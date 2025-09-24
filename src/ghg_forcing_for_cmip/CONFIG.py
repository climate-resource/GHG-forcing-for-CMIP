"""
In this module all global configurations for the data pipeline are stored
"""

import numpy as np
from prefect.cache_policies import INPUTS, RUN_ID, TASK_SOURCE

CACHE_POLICIES = TASK_SOURCE + INPUTS + RUN_ID

"""policies for task caching"""

GRID_CELL_SIZE = 5
"""size of single earth grid cell in degrees"""

# earth grid: latitudes
LAT_BIN_BOUNDS = np.arange(-90, 91, GRID_CELL_SIZE)
LAT_BIN_CENTRES = (LAT_BIN_BOUNDS[:-1] + LAT_BIN_BOUNDS[1:]) / 2

# earth grid: longitudes
LON_BIN_BOUNDS = np.arange(-180, 181, GRID_CELL_SIZE)
LON_BIN_CENTRES = (LON_BIN_BOUNDS[:-1] + LON_BIN_BOUNDS[1:]) / 2
