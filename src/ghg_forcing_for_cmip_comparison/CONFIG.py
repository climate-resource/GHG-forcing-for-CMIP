"""
global configuration file
"""

import numpy as np
from prefect.cache_policies import INPUTS, TASK_SOURCE

CACHE_POLICIES = TASK_SOURCE + INPUTS
"""policies for task caching"""

# earth grid: latitudes
LAT_BIN_BOUNDS = np.arange(-90, 91, 5)
LAT_BIN_CENTRES = (LAT_BIN_BOUNDS[:-1] + LAT_BIN_BOUNDS[1:]) / 2

# earth grid: longitudes
LON_BIN_BOUNDS = np.arange(-180, 181, 5)
LON_BIN_CENTRES = (LON_BIN_BOUNDS[:-1] + LON_BIN_BOUNDS[1:]) / 2

MIN_POINTS_FOR_SPATIAL_INTERPOLATION = 4
"""minimum number of observations required for grid cell to interpolate"""
