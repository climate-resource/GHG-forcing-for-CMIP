# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import requests
import zipfile
import io

from ghg_forcing_for_cmip_comparison import plotting

# %% [markdown]
# # Inspect binned data

# %%
url_ch4 = "https://github.com/climate-resource/GHG-forcing-for-CMIP-comparison/releases/download/v0.1.0-alpha/ch4.zip"
url_co2 = "https://github.com/climate-resource/GHG-forcing-for-CMIP-comparison/releases/download/v0.1.0-alpha/co2.zip"

dfs_ch4, dfs_co2 = ({}, {})

for url, dfs in zip([url_ch4, url_co2], [dfs_ch4, dfs_co2]):
    response = requests.get(url)

    # Open the zip file in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        # Loop through each file inside the zip
        for filename in thezip.namelist():
            if filename.endswith('binned.csv'):
                with thezip.open(filename) as file:
                    # Read CSV into pandas dataframe
                    dfs[filename] = pd.read_csv(file)

# %% [markdown]
# ## Methane ($CH_4$)

# %%
# load data set
d_binned = dfs_ch4["ch4_binned.csv"]

d_binned.head()

# %% [markdown]
# ### Inspect gridding
# + data are required to be on a $5^\circ \times 5^\circ$ grid

# %%
print("latitude:")
print(np.sort(d_binned.lat.unique()))
print("longitude:")
print(np.sort(d_binned.lon.unique()))

# %%
plotting.plot_map_grid(d_binned)

# %% [markdown]
# ## Carbon Dioxide ($CO_2$)

# %%
# load data set
d_binned = dfs_co2["co2_binned.csv"]

d_binned.head()

# %%
print("latitude:")
print(np.sort(d_binned.lat.unique()))
print("longitude:")
print(np.sort(d_binned.lon.unique()))

# %%
plotting.plot_map_grid(d_binned)
