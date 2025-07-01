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

from ghg_forcing_for_cmip_comparison import plotting

# %% [markdown]
# # Inspect interpolated data
# ## Methane ($CH_4$)

# %% [markdown]
# ## Carbon Dioxide ($CO_2$)

# %%
# load data sets
path = "../../data/downloads/co2/"
d_interpol = pd.read_csv(path + "co2_interpolated.csv")
d_binned = pd.read_csv(path + "co2_binned.csv")

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
np.sort(d_interpol[(d_interpol.year == 2007) & (d_interpol.month == 1)].lon.unique())  # noqa: PLR2004

# %%
plotting.plot_map_interpolated(d_interpol, d_binned, year=2007, month=1, gas="co2")

# %%
plotting.plot_map_interpolated(d_interpol, d_binned, year=2017, month=1)

# %%
plotting.plot_annual_concentration(d_interpol, gas="co2")
