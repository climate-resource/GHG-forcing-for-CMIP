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
from ghg_forcing_for_cmip_comparison.utils import load_data

# %% [markdown]
# # Inspect interpolated data

# %% [markdown]
# ## Methane ($CH_4$)

# %%
# load data sets
d_interpol = load_data("ch4_interpolated.csv")
d_binned = load_data("ch4_binned.csv")

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
d_interpol = load_data("co2_interpolated.csv")
d_binned = load_data("co2_binned.csv")

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
