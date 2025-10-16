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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghg_forcing_for_cmip import download_ground_based, download_satellite, plotting
from ghg_forcing_for_cmip.validation import compute_discrepancy_collocated

# run pipeline to download data
for gas in ["ch4", "co2"]:
    download_ground_based.download_surface_data(gas=gas, remove_original_files=True)
    download_satellite.download_satellite_data(gas=gas, remove_original_files=True)
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
_, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(11, 6))
plotting.plot_map(
    d_gb_ch4,
    f"NOAA observations sites ({d_gb_ch4.year.min()}-{d_gb_ch4.year.max()})"
    + "\n $CH_4$ data",
    axs[0],
)
plotting.plot_map(
    d_gb_co2,
    f"NOAA observations sites ({d_gb_co2.year.min()}-{d_gb_co2.year.max()})"
    + "\n $CO_2$ data",
    axs[1],
)

# %% [markdown]
# ### Satellite data
# #### Methane

# %%
_, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(9, 3))
for i, year in enumerate([2003, 2008, 2013]):
    plotting.plot_map(
        d_eo_ch4[d_eo_ch4.year == year],
        f"OBS4MIPs ({year}), $CH_4$ data",
        axs[0, i],
        "lon",
        "lat",
        ".",
        5,
    )
for i, year in enumerate([2018, 2020, 2022]):
    plotting.plot_map(
        d_eo_ch4[d_eo_ch4.year == year],
        f"OBS4MIPs ({year}), $CH_4$ data",
        axs[1, i],
        "lon",
        "lat",
        ".",
        5,
    )

# %% [markdown]
# #### Carbon Dioxide

# %%
_, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(9, 3))
for i, year in enumerate([2003, 2008, 2013]):
    plotting.plot_map(
        d_eo_co2[d_eo_co2.year == year],
        f"OBS4MIPs ({year}), $CO_2$ data",
        axs[0, i],
        "lon",
        "lat",
        ".",
        5,
    )
for i, year in enumerate([2018, 2020, 2022]):
    plotting.plot_map(
        d_eo_co2[d_eo_co2.year == year],
        f"OBS4MIPs ({year}), $CO_2$ data",
        axs[1, i],
        "lon",
        "lat",
        ".",
        5,
    )

# %% [markdown]
# ## Comparison of raw ground-based data with satellite product

# %%
# select collocated sites
d_gb_ch4_sel = d_gb_ch4[
    ["year", "month", "lat", "lon", "value", "site_code"]
].drop_duplicates()
d_eo_ch4_sel = d_eo_ch4[
    ["year", "month", "lat", "lon", "value", "pre", "vmr_profile_apriori"]
].drop_duplicates()
d_gb_co2_sel = d_gb_co2[
    ["year", "month", "lat", "lon", "value", "site_code"]
].drop_duplicates()
d_eo_co2_sel = d_eo_co2[
    ["year", "month", "lat", "lon", "value", "pre", "vmr_profile_apriori"]
].drop_duplicates()

d_colloc_ch4 = d_gb_ch4_sel.merge(
    d_eo_ch4_sel,
    on=["year", "month", "lat", "lon"],
    how="inner",
    suffixes=["_gb", "_eo"],
)
d_colloc_co2 = d_gb_co2_sel.merge(
    d_eo_co2_sel,
    on=["year", "month", "lat", "lon"],
    how="inner",
    suffixes=["_gb", "_eo"],
)

d_colloc_ch4.to_csv("data/downloads/ch4/ch4_collocated.csv")
d_colloc_co2.to_csv("data/downloads/co2/co2_collocated.csv")
d_colloc_ch4.head()

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 3), constrained_layout=True)
plotting.plot_monthly_average(d_colloc_ch4, "$CH_4$", axs[0])
plotting.plot_monthly_average(d_colloc_co2, "$CO_2$", axs[1])
fig.suptitle("Annual, global average of GHG concentration")

# %%
plotting.plot_average_hemisphere(d_colloc_ch4, "$CH_4$")

# %%
plotting.plot_average_hemisphere(d_colloc_co2, "$CO_2$")

# %%
d_col_sorted_co2 = (
    d_colloc_co2.groupby(["lat", "lon", "site_code"])
    .agg({"value_gb": "count"})
    .reset_index()
    .sort_values(by="value_gb", ascending=False)
)
d_col_sorted_ch4 = (
    d_colloc_ch4.groupby(["lat", "lon", "site_code"])
    .agg({"value_gb": "count"})
    .reset_index()
    .sort_values(by="value_gb", ascending=False)
)
d_col_sorted_ch4.rename(columns={"value_gb": "count"}).head(10)

# %%
max_lat = -50.0
d_col_sorted_co2[d_col_sorted_co2["lat"] <= max_lat].rename(
    columns={"value_gb": "count"}
).head(2)
# d_col_sorted_ch4[d_col_sorted_ch4["lat"] <= max_lat].rename(
#     columns={"value_gb": "count"}
# ).head(2)

# %%
min_lat = 50.0
d_col_sorted_co2[d_col_sorted_co2["lat"] >= max_lat].rename(
    columns={"value_gb": "count"}
).head(2)
# d_col_sorted_ch4[d_col_sorted_ch4["lat"] >= max_lat].rename(
#     columns={"value_gb": "count"}
# ).head(2)

# %%
sites_selected_co2 = [
    "SMO",
    "MLO",
    # "TAP",
    # "CGO",
    # "NMB",
    "USH",
    "OXK",
]
sites_selected_ch4 = [
    "SMO",
    "MLO",
    # "WLG",
    # "CGO",
    # "ASK",
    "USH",
    "BRW",
]

_, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 5))
# Could we add a label argument here
# so that each point is labelled with that label
# (and has a different marker)
# so then we can add a legend and see which site is which?
plotting.plot_map(
    d_col_sorted_co2[d_col_sorted_co2.site_code.isin(sites_selected_co2)],
    "Selected sites - CO2",
    lon_value="lon",
    lat_value="lat",
    axs=axs[0],
)
plotting.plot_map(
    d_col_sorted_ch4[d_col_sorted_ch4.site_code.isin(sites_selected_ch4)],
    "Selected sites - CH4",
    lon_value="lon",
    lat_value="lat",
    axs=axs[1],
)

# %%
d_col_sel_co2 = d_colloc_co2[
    d_colloc_co2.site_code.isin(sites_selected_co2)
].reset_index(drop=True)
d_col_sel_ch4 = d_colloc_ch4[
    d_colloc_ch4.site_code.isin(sites_selected_ch4)
].reset_index(drop=True)

d_col_sel_co2.to_csv("data/downloads/co2/co2_collocated_sites.csv")
d_col_sel_ch4.to_csv("data/downloads/ch4/ch4_collocated_sites.csv")

# %%
fig, axs = plt.subplots(2, 4, figsize=(11, 4), constrained_layout=True)
for i, site in enumerate(sites_selected_co2):
    d = d_col_sel_co2[d_col_sel_co2.site_code == site].copy()
    rmse_co2 = compute_discrepancy_collocated(d, "co2", "rmse")
    axs[0, i] = plotting.plot_monthly_average(d, "$CO_2$", axs[0, i])
    axs[0, i].set_title(
        site
        + f", RMSE: {rmse_co2['rmse_co2'].values[0]:.2f},"
        + f"\nBias: {rmse_co2['bias_co2'].values[0]:.2f}, "
        + f"SD: {np.sqrt(rmse_co2['var_co2'].values[0]):.2f}",
        fontsize="small",
    )
    axs[0, i].legend(fontsize="x-small", handlelength=0.4)

for i, site in enumerate(sites_selected_ch4):
    d = d_col_sel_ch4[d_col_sel_ch4.site_code == site].copy()
    rmse_ch4 = compute_discrepancy_collocated(d, "ch4", "rmse")
    axs[1, i] = plotting.plot_monthly_average(d, "$CH_4$", axs[1, i])
    axs[1, i].set_title(
        site
        + f", RMSE: {rmse_ch4['rmse_ch4'].values[0]:.2f},"
        + f"\nBias: {rmse_ch4['bias_ch4'].values[0]:.2f}, "
        + f"SD: {np.sqrt(rmse_ch4['var_ch4'].values[0]):.2f}",
        fontsize="small",
    )
    axs[1, i].legend(fontsize="x-small", handlelength=0.4)

# %%
plotting.plot_collocated_rmse(d_colloc_co2, d_colloc_ch4, "rmse")
