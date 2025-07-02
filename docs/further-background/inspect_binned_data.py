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

# %% jupyter={"outputs_hidden": true}
import numpy as np
import pandas as pd

from ghg_forcing_for_cmip_comparison import plotting

# %% [markdown]
# # Inspect binned data
# ## Methane ($CH_4$)

# %%
# load data sets
path = "../../data/downloads/ch4/"
d_binned = pd.read_csv(path + "ch4_binned.csv")

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
# load data sets
path = "../../data/downloads/co2/"
d_binned = pd.read_csv(path + "co2_binned.csv")

d_binned.head()

# %%
print("latitude:")
print(np.sort(d_binned.lat.unique()))
print("longitude:")
print(np.sort(d_binned.lon.unique()))

# %%
plotting.plot_map_grid(d_binned)
