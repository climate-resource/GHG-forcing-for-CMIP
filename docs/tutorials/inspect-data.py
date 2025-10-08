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
import pandas as pd
import plotting

# %%
d_gb_ch4 = pd.read_csv("../../data/downloads/ch4/ch4_gb_raw.csv")
d_gb_co2 = pd.read_csv("../../data/downloads/co2/co2_gb_raw.csv")
d_gb_ch4.head()

# %%
d_eo_ch4 = pd.read_csv("../../data/downloads/ch4/ch4_eo_raw.csv")
d_eo_co2 = pd.read_csv("../../data/downloads/co2/co2_eo_raw.csv")
d_eo_ch4.head()

# %% [markdown]
# ## Coverage

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
# ## Comparison of raw ground-based data with satellite product

# %%
# select collocated sites
d_gb_ch4_sel = d_gb_ch4[["year", "month", "lat", "lon", "value"]].drop_duplicates()
d_eo_ch4_sel = d_eo_ch4[
    ["year", "month", "lat", "lon", "value", "pre", "vmr_profile_apriori"]
].drop_duplicates()
d_gb_co2_sel = d_gb_co2[["year", "month", "lat", "lon", "value"]].drop_duplicates()
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
d_colloc_ch4.head()

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 3), constrained_layout=True)
plotting.plot_monthly_average(d_colloc_ch4, "$CH_4$", axs[0])
plotting.plot_monthly_average(d_colloc_co2, "$CO_2$", axs[1])
fig.suptitle("Annual, global average of GHG concentration")

# %%
plotting.plot_average_hemisphere(
    d_colloc_ch4[d_colloc_ch4.value_gb > 1600].copy(),  # noqa: PLR2004
    "$CH_4$",
)

# %%
plotting.plot_average_hemisphere(
    d_colloc_co2[d_colloc_co2.value_gb > 360].copy(),  # noqa: PLR2004
    "$CO_2$",
)

# %% [markdown]
# ## Inspect vertical dimension of satellite data

# %%
fig, axs = plt.subplots(1, 2, constrained_layout=True, sharey=True, figsize=(6, 4))
plotting.plot_vertical_apriori(d_colloc_ch4, axs[0], "$CH_4$")
plotting.plot_vertical_apriori(d_colloc_co2, axs[1], "$CO_2$")
axs[0].set_ylabel("pressure-center \n (normalized to surface pressure)")
fig.suptitle("VMR apriori profile - OBS4MIPS (global avg.)")


# %%
def adjust_vertical(d: pd.DataFrame, pressure: float) -> pd.DataFrame:
    """
    Apply simple method to adjust vertical dimension

    Parameters
    ----------
    d :
        dataframe incl. vertical information

    pressure :
        pressure level according which should
        be normalized

    Returns
    -------
    :
        dataframe with adjusted ghg concentration value
    """
    d_adj = d[d.pre == pressure].reset_index().copy()
    d_adj["factor"] = d_adj.value_eo / d_adj.vmr_profile_apriori
    d_adj["value_eo"] = d_adj.value_eo * (1 / d_adj.factor)
    return d_adj


d_adj_ch4 = adjust_vertical(d_colloc_ch4, 0.75)
d_adj_co2 = adjust_vertical(d_colloc_co2, 0.95)

_, axs = plt.subplots(1, 2, figsize=(11, 3), constrained_layout=True)
plotting.plot_monthly_average(d_colloc_ch4, "$CH_4$", axs[0], label_gb=None)
plotting.plot_monthly_average(d_adj_ch4, "$CH_4$", axs[0], label_eo="satellite_adj")
plotting.plot_monthly_average(d_colloc_co2, "$CO_2$", axs[1], label_gb=None)
plotting.plot_monthly_average(d_adj_co2, "$CO_2$", axs[1], label_eo="satellite_adj")
