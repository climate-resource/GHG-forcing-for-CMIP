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
import matplotlib.pyplot as plt
import pandas as pd
import requests
import zipfile
import io

from ghg_forcing_for_cmip_comparison import plotting

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Inspect raw data

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
            if filename.endswith('raw.csv'):
                with thezip.open(filename) as file:
                    # Read CSV into pandas dataframe
                    dfs[filename] = pd.read_csv(file)

# %% [markdown]
# ## Methane ($CH_4$)

# %%
# load data set
d_raw = dfs_ch4["ch4_raw.csv"]

d_raw.head()

# %%
d_raw.columns

# %%
d = d_raw.groupby(["year", "month"]).agg({"value": "mean", "time_fractional": "mean"})

plt.figure(figsize=(7, 3))
plt.plot(d.time_fractional, d.value)
plt.title("monthly average of raw CH4 data")
plt.xlabel("year/month")
plt.ylabel("ppb")
plt.show()

# %% [markdown]
# ### Observation networks
# #### AGAGE / GAGE
#
# The Advanced Global Atmospheric Gases Experiment (AGAGE: 1993–present) and its predecessors (Atmospheric Lifetime Experiment, ALE: 1978–1981; Global Atmospheric Gases Experiment, GAGE: 1982–1992) have measured the greenhouse gas and ozone-depleting gas composition of the global atmosphere continuously since 1978.
#
# + The ALE program was instigated to measure the then five major ozone-depleting gases (CFC-11 (CFCl3), CFC-12 (CCl2F2), CCl4, CH3CCl3, N2O) in the atmosphere four times per day using automated gas chromatographs with electron-capture detectors (GC-ECDs) at four stations around the globe and to determine the atmospheric lifetimes of the purely anthropogenic of these gases from their measurements and industry data on their emissions (Prinn et al., 1983a).
# + The GAGE project broadened the global coverage to five stations, the number of gases being measured to eight (adding CFC-113 (CCl2FCClF2), CHCl3, and CH4 to the ALE list), and the frequency to 12 per day by improving the GC-ECDs and adding gas chromatographs with flame-ionization detectors (GC-FIDs; Prinn et al., 2000).
# + The AGAGE program then significantly improved upon the GAGE instruments by increasing their measurement precision and frequency (to 36 per day) and adding gas chromatographs with mercuric oxide reduction detectors, to measure 10 biogenic and/or anthropogenic gases overall (adding H2 and CO to the GAGE list).
#
# #### AGAGE (5 stations):
# + Cape Matatula, American Samoa (14° S, 171° W; 77 m, 1978 to present)
# + Trinidad Head, California (41° N, 124° W; 140 m, 1995 to present)
# + Ragged Point, Barbados (13° N, 59° W; 42 m, 1978 to present)
# + Mace Head (53° N, 10° W; 25 m, 1987 to present)
# + Cape Grim, Tasmania, Australia (41° S, 145° E; 164 m, 169 m, 1978 to present)
#
# Source:
# + [source information AGAGE/GAGE](https://essd.copernicus.org/articles/10/985/2018/)
# + [source information NOAA](https://gml.noaa.gov/dv/site/index.php?stacode=MKO)

# %%
d_AGAGE = (
    d_raw[d_raw.network == "agage"][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_AGAGE

# %%
plotting.plot_map(
    d=d_AGAGE,
    title=r"$CH_4$ - Advanced Global Atmospheric"
    + "\n Gases Experiment (AGAGE) stations",
)

# %% [markdown]
# #### GAGE (4 stations):
# + Cape Matatula, American Samoa (14° S, 171° W; 77 m, 1982 to present)
# + Trinidad Head, California (41° N, 124° W; 140 m, 1995 to present)
# + Mace Head (53° N, 10° W; 25 m, 1987 to present)
# + Cape Grim, Tasmania, Australia (41° S, 145° E; 164 m, 169 m, 1978 to present)

# %%
d_GAGE = (
    d_raw[d_raw.network == "gage"][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_GAGE

# %%
plotting.plot_map(
    d=d_GAGE,
    title=r"$CH_4$ - Global Atmospheric Gases" + "\nExperiment (GAGE) stations",
)

# %% [markdown]
# #### NOAA - insitu (3 stations):
# + Mauna Loa, Hawaii (19.5° N, 155.5° W; 3397.00 masl)
# + Mauna Kea, Hawaii (19.8° N, 155.4° W; 4199.00 masl)
# + Barrow Atmospheric Baseline Observatory, US (71.3° N, 156.6° W; 11.00 masl)

# %%
d_insitu = (
    d_raw[d_raw.insitu_vs_flask == "insitu"][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_insitu

# %%
plotting.plot_map(d=d_insitu, title=r"$CH_4$ - NOAA insitu stations")

# %% [markdown]
# #### NOAA - flask
# + thereof 6 ships and 274 surface measurements

# %%
d_flask = (
    d_raw[
        (d_raw.insitu_vs_flask == "flask")
        & (d_raw.sampling_strategy == "shipboard-flask")
    ][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_flask.site_code.unique()

# %%
plotting.plot_map(d=d_flask, title=r"$CH_4$ - NOAA flask: Shipboards")

# %%
d_flask_surf = (
    d_raw[
        (d_raw.insitu_vs_flask == "flask")
        & (d_raw.sampling_strategy == "surface-flask")
    ][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_flask_surf.site_code.unique()

# %%
plotting.plot_map(d=d_flask_surf, title=r"$CH_4$ - NOAA flask: Surface")

# %% [markdown]
# ## Carbon Dioxide ($CO_2$)

# %%
# load data set
d_raw = dfs_co2["co2_raw.csv"]

d_raw.head()

# %%
d_raw.columns

# %%
d = d_raw.groupby(["year", "month"]).agg({"value": "mean", "time_fractional": "mean"})

plt.figure(figsize=(7, 3))
plt.plot(d.time_fractional, d.value)
plt.title("monthly average of raw CO2 data")
plt.xlabel("year/month")
plt.ylabel("ppm")
plt.show()

# %% [markdown]
# ### NOAA insitu (5 stations)
# + Cape Matatula, American Samoa (14°S, 171°W; 77m)
# + Mauna Loa, Hawaii (19.5° N, 155.5° W; 3397.00 masl)
# + Mauna Kea, Hawaii (19.8° N, 155.4° W; 4199.00 masl)
# + Barrow Atmospheric Baseline Observatory, US (71.3° N, 156.6° W; 11.00 masl)
# + Halley Station, Antarctica, UK (75.5° S, 25.6° W; 30.00 masl)

# %%
d_insitu = (
    d_raw[d_raw.insitu_vs_flask == "insitu"][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_insitu

# %%
plotting.plot_map(d=d_insitu, title=r"$CO_2$ - NOAA insitu stations")

# %% [markdown]
# ### NOAA - flask

# %%
d_flask = (
    d_raw[
        (d_raw.insitu_vs_flask == "flask")
        & (d_raw.sampling_strategy == "shipboard-flask")
    ][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_flask.site_code.unique()


# %%
plotting.plot_map(d=d_flask, title=r"$CO_2$ - NOAA flask: Shipboards")

# %%
d_flask_surf = (
    d_raw[
        (d_raw.insitu_vs_flask == "flask")
        & (d_raw.sampling_strategy == "surface-flask")
    ][["site_code", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_flask_surf.site_code.unique()


# %%
plotting.plot_map(d=d_flask_surf, title=r"$CO_2$ - NOAA flask: Surface")
