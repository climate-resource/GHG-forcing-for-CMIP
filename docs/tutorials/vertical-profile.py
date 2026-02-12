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
# # Model vertical profile

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghg_forcing_for_cmip import plotting
from ghg_forcing_for_cmip.validation import compute_discrepancy_collocated

# %%
# ground-based observations
d_gb_ch4 = pd.read_csv("data/downloads/ch4/ch4_gb_raw.csv")
d_gb_co2 = pd.read_csv("data/downloads/co2/co2_gb_raw.csv")
# satellite products
d_eo_ch4 = pd.read_csv("data/downloads/ch4/ch4_eo_raw.csv")
d_eo_co2 = pd.read_csv("data/downloads/co2/co2_eo_raw.csv")

## collocated sites
dcol_co2 = pd.read_csv("data/downloads/co2/co2_collocated_sites.csv")
dcol_ch4 = pd.read_csv("data/downloads/ch4/ch4_collocated_sites.csv")


# %% jupyter={"source_hidden": true}
def get_d_vert(dcol: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data for inspecting vertical dimension

    Parameters
    ----------
    dcol :
        dataframe with collocated sites

    Returns
    -------
    :
        aggregated dataframe
    """
    d_vert = (
        dcol.groupby(["pre", "site_code"])
        .agg(
            {
                "vmr_profile_apriori": ("mean", "std"),
                "value_gb": ("mean", "std"),
                "value_eo": ("mean", "std"),
            }
        )
        .reset_index()
    )

    d_vert.columns = [
        "_".join([a, b]) if len(b) > 1 else a for (a, b) in d_vert.columns
    ]
    return d_vert


d_vert_co2 = get_d_vert(dcol_co2)
d_vert_ch4 = get_d_vert(dcol_ch4)


# %% jupyter={"source_hidden": true}
def plot_vertical(d: pd.DataFrame, gas: str) -> None:
    """
    Plot vertical dimension for collocated sites

    vertical profile is taken from apriori profile
    of OBS4MIPs data, which is based on CarbonTracker.

    Parameters
    ----------
    d :
        aggregated dataframe with information about vertical
        dimension

    gas :
        target greenhouse gas (either co2 or ch4)

    """
    pre_labels = np.round(np.subtract(1, d.pre.unique()), 2).astype(str)
    pre_labels = np.where(pre_labels == "0.05", "top", pre_labels)
    pre_labels = np.where(pre_labels == "0.95", "surf", pre_labels)

    _, axs = plt.subplots(1, 4, constrained_layout=True, sharey=True, figsize=(7, 3))
    for i, site in enumerate(d.site_code.unique()):
        d_sel = d[d.site_code == site]
        axs[i].plot(d_sel.vmr_profile_apriori_mean, 1 - d_sel.pre, "-o")
        axs[i].fill_betweenx(
            1 - d_sel.pre,
            d_sel.vmr_profile_apriori_mean - d_sel.vmr_profile_apriori_std,
            d_sel.vmr_profile_apriori_mean + d_sel.vmr_profile_apriori_std,
            color="skyblue",
            alpha=0.4,
        )
        axs[i].axvline(
            d_sel.vmr_profile_apriori_mean.mean(), color="orange", label="apriori"
        )
        axs[i].axvline(d_sel.value_gb_mean.unique(), color="darkred", label="GB")
        axs[i].axvline(
            d_sel.value_eo_mean.unique(), linestyle="--", color="green", label="EO"
        )
        axs[i].set_title(site)
        axs[i].set_xlabel(rf"${'_'.join([gas[:2].upper(), gas[-1]])}$ (apriori)")
    axs[0].set_yticks(d.pre.unique(), pre_labels)
    axs[0].set_ylabel("pressure (center)")
    axs[0].legend(handlelength=0.6, frameon=False)


# %%
plot_vertical(d_vert_co2, "co2")

# %%
plot_vertical(d_vert_ch4, "ch4")

# %% jupyter={"source_hidden": true}
fig, axs = plt.subplots(2, 4, figsize=(11, 4), constrained_layout=True)

# Same zip trick as above
for site, ax in zip(dcol_co2.site_code.unique(), axs[0, :]):
    d = dcol_co2[dcol_co2.site_code == site].copy()
    rmse_co2 = compute_discrepancy_collocated(d, "co2", "rmse")
    ax = plotting.plot_monthly_average(d, "$CO_2$", ax)  # noqa: PLW2901
    ax.set_title(
        # As above re ability to use brackets instead of +
        # (although this is really very optional
        # and probably more a style choice than there being a right answer)
        site
        + f", RMSE: {rmse_co2['rmse_co2'].values[0]:.2f},"
        + f"\nBias: {rmse_co2['bias_co2'].values[0]:.2f}, "
        + f"SD: {np.sqrt(rmse_co2['var_co2'].values[0]):.2f}",
        fontsize="small",
    )
    ax.legend(fontsize="x-small", handlelength=0.4, frameon=False)

# zip could be used here too
for i, site in enumerate(dcol_ch4.site_code.unique()):
    d = dcol_ch4[dcol_ch4.site_code == site].copy()
    rmse_ch4 = compute_discrepancy_collocated(d, "ch4", "rmse")
    axs[1, i] = plotting.plot_monthly_average(d, "$CH_4$", axs[1, i])
    axs[1, i].set_title(
        site
        + f", RMSE: {rmse_ch4['rmse_ch4'].values[0]:.2f},"
        + f"\nBias: {rmse_ch4['bias_ch4'].values[0]:.2f}, "
        + f"SD: {np.sqrt(rmse_ch4['var_ch4'].values[0]):.2f}",
        fontsize="small",
    )
    axs[1, i].legend(fontsize="x-small", handlelength=0.4, frameon=False)

# %%
