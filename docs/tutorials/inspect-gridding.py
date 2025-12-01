# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inspect gridding
#

# %%
import pandas as pd

from ghg_forcing_for_cmip import CONFIG, plotting

# %%
# load raw data set
df_ch4 = pd.read_csv("data/downloads/ch4/ch4_gb_raw.csv")
df_ch4_eo = pd.read_csv("data/downloads/ch4/ch4_eo_raw.csv")
df_co2 = pd.read_csv("data/downloads/co2/co2_gb_raw.csv")
df_co2_eo = pd.read_csv("data/downloads/co2/co2_eo_raw.csv")

# %% [markdown]
# ## Inspect gridding and coverage for ground-based data

# %%
# plot subset of entire grid
# shows raw data points and grid-average value
plotting.plot_gridsizes(df_ch4, CONFIG.GRID_CELL_SIZE, True)

# %%
# plot entire grid
# shows only grid-average values
plotting.plot_gridsizes(df_ch4, CONFIG.GRID_CELL_SIZE, False)

# %%
# plot subset of entire grid
# shows raw data points and grid-average value
plotting.plot_gridsizes(df_co2, CONFIG.GRID_CELL_SIZE, True)

# %%
# plot entire grid
# shows only grid-average values
plotting.plot_gridsizes(df_co2, CONFIG.GRID_CELL_SIZE, False)

# %% [markdown]
# ## Inspect coverage for ground-based vs. satellite data
# + show only subset of grid for better visualization
# + ground-based data are not adjusted for vertical dimension therefore we
# see clear differences in the value between ground-based and satellite observations
#
# Satellite product would clearly improve the coverage information.
# Question is how to assimilate satellite and ground-based data given
# that both products measure different things?

# %%
plotting.plot_coverage(df_ch4_eo, df_ch4, CONFIG.GRID_CELL_SIZE)

# %%
plotting.plot_coverage(df_co2_eo, df_co2, CONFIG.GRID_CELL_SIZE)
