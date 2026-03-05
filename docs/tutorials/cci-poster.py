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
# # Inspect data
#

# %%
import logging

import matplotlib.pyplot as plt
import pandas as pd

from ghg_forcing_for_cmip import plotting

# silent prefect run
logging.getLogger("prefect").setLevel(logging.ERROR)

# # run pipeline to download data
# for gas in ["co2", "ch4"]:
#     download_ground_based.download_surface_data(gas=gas, remove_original_files=False)
#     download_satellite.download_satellite_data(gas=gas, remove_original_files=False)
# %%
d_gb_ch4 = pd.read_csv("data/downloads/ch4/ch4_gb_raw.csv")
d_gb_co2 = pd.read_csv("data/downloads/co2/co2_gb_raw.csv")
d_gb_ch4.head()

# %%
d_eo_ch4 = pd.read_csv("data/downloads/ch4/ch4_eo_raw.csv")
d_eo_co2 = pd.read_csv("data/downloads/co2/co2_eo_raw.csv")
d_eo_ch4.head()

# %% [markdown]
# ## Coverage
# ### Ground-based data

# %%
fig, ax = plt.subplots(figsize=(9, 3))
plotting.plot_map(
    d_gb_co2,
    title=f"$CO_2$ surface coverage ({d_gb_co2.year.min()}-{d_gb_co2.year.max()})",
    axs=ax,
)
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])

# %% [markdown]
# ### Satellite data

# %%
year = 2022
fig, ax = plt.subplots(figsize=(9, 3))
plotting.plot_map(
    d_eo_co2[d_eo_co2.year == year],
    title=f"$CO_2$ satellite coverage from Obs4MIPs ({year})",
    axs=ax,
    lon_value="lon",
    lat_value="lat",
    marker=".",
    markersize=10,
)
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
