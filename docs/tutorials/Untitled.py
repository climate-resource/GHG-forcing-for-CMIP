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
import pmdarima as pma
import pandas as pd
import numpy as np
import bambi as bmb
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas
from geodatasets import get_path
import pymc as pm
import statsmodels

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from ghg_forcing_for_cmip import CONFIG, plotting, utils
from itertools import product

# %%
df_ch4_eo = pd.read_csv("../../data/downloads/co2/co2_eo_raw.csv")
df_ch4 = pd.read_csv("../../data/downloads/co2/co2_gb_raw.csv")

# %%
# template dataframe with complete information for every 
# year x month x lat x lon combination
df_template = pd.DataFrame(
    product(df_ch4.year.unique(), np.arange(1,13,1), 
            df_ch4.lat.unique(), df_ch4.lon.unique()),
    columns = ["year", "month", "lat", "lon"]
)

# combine ground-based and satellite data into one dataset
sel_cols = ["year", "month", "lat", "lon", "value"]

df = (df_ch4[sel_cols]
 .drop_duplicates()
 .merge(
     df_ch4_eo[sel_cols],
     on=["year", "month", "lat", "lon"],
     how="outer",
     suffixes=("_gb", "_eo")
 )
)

df = df.merge(df_template, on=["year", "month", "lat", "lon"], how="outer")

df['decimal_year'] = df['year'] + (df['month'] - 1) / 12
df["day"] = 1
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df = df.drop(columns=["year", "month", "day", "decimal_year"]).set_index("date")
df.head()

# %%
no_total = len(df)
no_missing_both = len(df[(df.value_gb.isna()) & (df.value_eo.isna())])
no_missing_gb = len(df[(df.value_gb.isna()) & (~df.value_eo.isna())])
no_missing_eo = len(df[(~df.value_gb.isna()) & (df.value_eo.isna())])

print("Both GB and EO are missing: ", 
      no_missing_both, f"({no_missing_both/no_total:.2f}%)")
print("Only GB is missing: ", 
      no_missing_gb, f"({no_missing_gb/no_total:.2f}%)")
print("Only EO is missing: ", 
      no_missing_eo, f"({no_missing_eo/no_total:.2f}%)")

# %%
# collocated dataset
df_all = (df_ch4[sel_cols]
 .drop_duplicates()
 .merge(
     df_ch4_eo[sel_cols],
     on=["year", "month", "lat", "lon"],
     how="outer",
     suffixes=("_gb", "_eo")
 )
)
df_coll = (
    df_all[(~df_all.value_gb.isna()) & (~df_all.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)
df_coll["day"] = 15
df_coll["date"] = pd.to_datetime(df_coll[["year", "month", "day"]])
df_coll = df_coll.drop(columns= "day").drop_duplicates().set_index("date")
df_coll

# %%
prior_hsgp = {
    "sigma": bmb.Prior("Exponential", lam=2), # amplitude
    "ell": bmb.Prior("InverseGamma", mu=10, sigma=1) # lengthscale
}

# This is the dictionary we pass to Bambi
priors = {
    "hsgp(x, m=10, c=2)": prior_hsgp,
    "sigma": bmb.Prior("HalfNormal", sigma=10)
}

model1 = bmb.Model("value_gb ~ 1 + year + (1 | month) + (1 | lat) + lat + lon + value_eo", df_coll, categorical=["lat", "month"])
model2 = bmb.Model("value_gb ~ 1 + year + (1 | month) + (1 | lat) + hsgp(value_eo, m=10, c=2)", df_coll, categorical=["lat", "month"])
model3 = bmb.Model("value_gb ~ 0 + hsgp(value_eo, m=10, c=2)", df_coll, priors=priors)

# %%
idata = model3.fit(draws=2000, random_seed=42)

# %%
print(idata.sample_stats["diverging"].sum().to_numpy())

# %% jupyter={"outputs_hidden": true}
az.summary(idata)

# %%
model.predict(idata, kind="response")
df_coll["value_gb_pred"] = idata.posterior_predictive.value_gb.mean(dim=("chain", "draw")).values
df_coll.head()

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
df_coll["trend"] = idata.posterior.Intercept.mean().values + idata.posterior.year.mean().values * df_coll.year + idata.posterior.value_eo.mean().values * df_coll.value_eo
df_coll_month = df_coll.reset_index().groupby(["month"]).agg({"trend": "mean"}).reset_index() 
df_coll_month["season_offset"] = idata.posterior["1|month"].mean(dim=("chain", "draw")).values
df_coll_month["seasonality"] = df_coll_month["trend"] + df_coll_month["season_offset"]

#df_coll["seasonality"] = idata.posterior.Intercept.mean().values + idata.posterior.year.mean().values * df_coll.year + idata.posterior.month.mean().values + idata.posterior.value_eo.mean().values * df_coll.value_eo

df_trend = df_coll.groupby("year").agg({"trend":"mean"}).reset_index()
#plt.plot(df_trend.year, df_trend.trend)

#sns.lineplot(data=df_coll_month, x="month", y="seasonality")

# %%
conditions = [
    (df_coll['lat'] > 45),
    (df_coll['lat'] < -45)
]
choices = ['N', 'S']
df_coll['hemisphere'] = np.select(conditions, choices, default='Tropics')
df_coll_agg = df_coll.reset_index().groupby(["date", "hemisphere"]).agg({"value_gb": "mean", "value_eo":"mean", "value_gb_pred":"mean"}).reset_index()

# %%
fig, axs = plt.subplots(1,3, figsize=(10,3), constrained_layout=True)
for i, (hemi, name) in enumerate(zip(["S", "N","Tropics"], ["Southern Hemisphere", "Northern Hemisphere", "Tropics"])):
    sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere==hemi], x="date", y="value_gb", lw=2, ax=axs[i], label="GB")
    sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere==hemi], x="date", y="value_eo", lw=2, ax=axs[i], label="EO")
    sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere==hemi], x="date", y="value_gb_pred", lw=2, ax=axs[i], label="GB-pred")
    
    axs[i].set_title(name);

# %%
X = df_coll[["year", "month", "lat", "lon", "value_eo"]]
y = df_coll.value_gb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = X_train
df_train["value_gb"] = y_train
df_train.head()

# %%
X = df_coll[["lat", "lon", "value_eo"]]
y = df_coll.value_gb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
regr.fit(X_train, y_train)

# %%
mse = (regr.predict(X_test) - y_test)**2

plt.plot(mse, "o")

# %%
# select dataframe for which value_eo exist but not value_gb
df_eo_only = (
    df_all[(df_all.value_gb.isna()) & (~df_all.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)

df_eo_only["value_gb_pred"] = regr.predict(df_eo_only[["lat", "lon", "value_eo"]])
df_eo_only


# %%
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
# collocated dataset
df_coll = df[(~df.value_gb.isna()) & (~df.value_eo.isna())]

# calibration model
features = sel_cols[:-1] + ["value_eo"]
y = df_coll.value_gb
X = df_coll[features]

# Split for validation (to check if our filling is accurate)
tscv = TimeSeriesSplit(
    n_splits=5, 
    gap=48,
    max_train_size=10000,
    test_size=1000
)

gbr = GradientBoostingRegressor(random_state=42)
dt_reg = DecisionTreeRegressor(random_state=42)
mod = AutoReg(endog=y, lags=3, exog=X)
res = mod.fit()
