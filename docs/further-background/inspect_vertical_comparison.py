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
# # Inspect vertical distribution for ground-based data and final comparison with satellite data
# ## Methane ($CH_4$)

# %%
# load data sets
path = "data/downloads/ch4/"
d_interpol = pd.read_csv(path+"ch4_interpolated.csv")
d_vertical = pd.read_csv(path+"ch4_vertical.csv")
d_compare = pd.read_csv(path+"ch4_joint_comparison.csv")

d_compare.head()

# %% [markdown]
# ### Inspect vertical distribution of ground-based data

# %%
plotting.plot_vertical(
    d_vertical, gas="ch4", lat=2.5, p_surface=0.95, p_top=0.05
)

# %%
plotting.plot_annual_concentration_comparison(
    d_vertical, d_interpol, "ch4"
)

# %% [markdown]
# ### Inspect ground-based data filtered according to satellite observations

# %%
# value-column refers to ground-based data
# (with vertical dimension but without AK)
d_gb_modelled = d_compare.copy()
d_gb_modelled.rename(columns={"ground_based":"value"}, inplace=True)

plotting.plot_annual_concentration(
    d_gb_modelled, gas="ch4"
)

# %%
plotting.plot_map_combined(
    d_gb_modelled,
    years=[2004, 2006, 2008, 2010],
    month=8,
    gas="ch4"
)

# %%
plotting.plot_map_combined(
    d_gb_modelled,
    years=[2015, 2018, 2020, 2022],
    month=8,
    gas="ch4"
)

# %% [markdown]
# ### Compare satellite data with ground-based data

# %%
d_gb_AK = d_compare.copy()
d_eo = d_compare.copy()
d_eo.rename(columns={"satellite":"value"}, inplace=True)
d_gb_AK.rename(columns={"ground_based_AK":"value"}, inplace=True)

plotting.plot_eo_gb_seasonal_annual(d_gb_modelled, d_gb_AK, d_eo, gas="ch4")

# %% [markdown]
# ## Carbon Dioxide ($CO_2$)

# %%
# load data sets
path = "data/downloads/co2/"
d_interpol = pd.read_csv(path+"co2_interpolated.csv")
d_vertical = pd.read_csv(path+"co2_vertical.csv")
d_compare = pd.read_csv(path+"co2_joint_comparison.csv")

d_compare.head()

# %% [markdown]
# ### Inspect vertical distribution of ground-based data

# %%
plotting.plot_vertical(
    d_vertical, gas="co2", lat=2.5, p_surface=0.95, p_top=0.05
)

# %%
plotting.plot_annual_concentration_comparison(
    d_vertical, d_interpol, "co2"
)

# %% [markdown]
# ### Inspect ground-based data filtered according to satellite data

# %%
# value-column refers to ground-based data
# (with vertical dimension but without AK)
d_gb_modelled = d_compare.copy()
d_gb_modelled.rename(columns={"ground_based":"value"}, inplace=True)

plotting.plot_annual_concentration(
    d_gb_modelled, gas="co2"
)

# %%
plotting.plot_map_combined(
    d_gb_modelled,
    years=[2004, 2006, 2008, 2010],
    month=8,
    gas="co2"
)

# %%
plotting.plot_map_combined(
    d_gb_modelled,
    years=[2015, 2018, 2020, 2022],
    month=8,
    gas="co2"
)

# %% [markdown]
# ### Compare satellite data with ground-based data

# %%
d_gb_AK = d_compare.copy()
d_eo = d_compare.copy()
d_eo.rename(columns={"satellite":"value"}, inplace=True)
d_gb_AK.rename(columns={"ground_based_AK":"value"}, inplace=True)

plotting.plot_eo_gb_seasonal_annual(d_gb_modelled, d_gb_AK, d_eo, gas="co2")
