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
# # Inspect binned data

# %% [markdown]
# ## Methane ($CH_4$)

# %%
# load data set
d_binned = load_data("ch4_binned.csv")

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
d_binned = load_data("co2_binned.csv")

d_binned.head()

# %%
print("latitude:")
print(np.sort(d_binned.lat.unique()))
print("longitude:")
print(np.sort(d_binned.lon.unique()))

# %%
plotting.plot_map_grid(d_binned)
