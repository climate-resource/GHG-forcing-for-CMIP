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
# # Inspect interpolated data

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
            if filename.endswith(('binned.csv', 'interpolated.csv')):
                with thezip.open(filename) as file:
                    # Read CSV into pandas dataframe
                    dfs[filename] = pd.read_csv(file)

# %% [markdown]
# ## Methane ($CH_4$)

# %%
# load data sets
d_interpol = dfs_ch4["ch4_interpolated.csv"]
d_binned = dfs_ch4["ch4_binned.csv"]

d_interpol.head()

# %%
# interpolation doesn't work for latitude = 87.5
# yields NaN for all years/month (don't know why)
print("latitude:")
print(np.sort(d_interpol.lat.unique()))
# interpolation works fine for all longitudes
print("longitude:")
print(np.sort(d_interpol.lon.unique()))

# %%
plotting.plot_map_interpolated(
    d_interpol, d_binned, year=2007, month=6, gas="ch4",
    vmin=None, vmax=None
)

# %%
plotting.plot_map_interpolated(
    d_interpol, d_binned, year=2017, month=6, gas="co2",
    vmin=None, vmax=None
)


# %%
plotting.plot_annual_concentration(
    d_interpol, gas="ch4"
)

# %% [markdown]
# ## Carbon Dioxide ($CO_2$)

# %%
# load data sets
d_interpol = dfs_co2["co2_interpolated.csv"]
d_binned = dfs_co2["co2_binned.csv"]

d_interpol.head()

# %%
# interpolation doesn't work for latitude = 87.5
# yields NaN for all years/month (don't know why)
print("latitude:")
print(np.sort(d_interpol.lat.unique()))
# interpolation works fine for all longitudes
print("longitude:")
print(np.sort(d_interpol.lon.unique()))

# %%
plotting.plot_map_interpolated(
    d_interpol, d_binned, year=2007, month=6, gas="co2",
    vmin=None, vmax=None
)

# %%
plotting.plot_map_interpolated(
    d_interpol, d_binned, year=2017, month=6, gas="co2"
)

# %%
plotting.plot_annual_concentration(
    d_interpol, gas="co2"
)
