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

# %%
import pandas as pd
import xarray as xr
import bambi as bmb
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

# %%
random_seed = 1234

d_eo = pd.read_csv("data/downloads/ch4/ch4_eo_raw.csv")
d_gb = xr.open_dataset("data/downloads/ch4/ch4_binned_q0.5.nc").to_dataframe().reset_index()

# %%
d_eo_sel = d_eo[["year", "month", "lat", "lon", "value"]].drop_duplicates()
d_gb_sel = d_gb[["year", "month", "lat", "lon", "value"]]

d_all = d_eo_sel.merge(d_gb_sel, on=["year", "month", "lat", "lon"], 
                       suffixes=["_eo", "_gb"]).dropna()
d_all_2020 = d_all.groupby(["month", "lat", "lon"]).agg({"value_eo": "mean", "value_gb": "mean"}).reset_index()
d_all_2020.head()

# %%
# model building
formula_m1 = "value_gb ~ value_eo + lat:lon + month"
m1 = bmb.Model(formula_m1, data=d_all_2020, family="gaussian", )

# model fitting
idata_m1 = m1.fit(random_seed=random_seed)

# %%
az.summary(idata_m1)

# %%
az.plot_trace(idata_m1);

# %%
m1.predict(idata_m1, kind="response")
d_all_2020["predictions"] = idata_m1.posterior_predictive.value_gb.mean(("chain", "draw")).values
d_all_2020


# %%
def categorize_lat(lat):
    if lat >=30:
        return "N30"
    elif (lat > 15) and (lat < 30):
        return "N15"
    elif (lat < -15) and (lat > -30):
        return "S15"
    elif lat < -30:
        return "S30"
    else:
        return "Tropics"

d_all_2020["region"] = d_all_2020["lat"].apply(categorize_lat)
d_lat_agg = d_all_2020.groupby(["month", "lat"]).agg({"predictions": "mean", "value_gb": "mean", "value_eo": "mean"}).reset_index()

_, axs = plt.subplots(1,2, sharey=True, sharex=True, figsize=(7,3))
sns.heatmap(d_lat_agg.pivot(index="lat", columns="month", values="predictions"), ax=axs[0], vmin=1600, vmax=2100)
sns.heatmap(d_lat_agg.pivot(index="lat", columns="month", values="value_gb"), ax=axs[1], vmin=1600, vmax=2100) 
#sns.lineplotdata=d_agg, x="month", y="value_gb", hue="region", linestyle="dashed", legend=False) 
#sns.lineplot(data=d_agg, x="month", y="value_eo", hue="region", linestyle="dotted", legend=False) 

# %%
d_lon_agg = d_all_2020.groupby(["month", "lon"]).agg({"predictions": "mean", "value_gb": "mean", "value_eo": "mean"}).reset_index()

_, axs = plt.subplots(1,2, sharey=True, sharex=True, figsize=(7,3))
sns.heatmap(d_lon_agg.pivot(index="lon", columns="month", values="predictions"), ax=axs[0], vmin=1600, vmax=2100)
sns.heatmap(d_lon_agg.pivot(index="lon", columns="month", values="value_gb"), ax=axs[1], vmin=1600, vmax=2100) 

# %%
d_all_2020[d_all_2020.month==3]
