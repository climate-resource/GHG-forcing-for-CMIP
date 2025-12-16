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
import arviz as az
import nutpie

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from ghg_forcing_for_cmip import CONFIG, plotting, utils
from itertools import product

SEED = 1211

# %%
gas = "ch4"
df_ch4_eo = pd.read_csv(f"../../data/downloads/{gas}/{gas}_eo_raw.csv")
df_ch4 = pd.read_csv(f"../../data/downloads/{gas}/{gas}_gb_raw.csv")

# %%
# target variables (used for grouping, selection, etc.)
sel_cols = ["year", "month", "lat", "lon"]

# template dataframe with complete information for every 
# year x month x lat x lon combination
df_template = pd.DataFrame(
    product(df_ch4.year.unique(), np.arange(1,13,1), 
            df_ch4.lat.unique(), df_ch4.lon.unique()),
    columns = sel_cols
)

# combine ground-based and satellite data into one dataset
df_combined = (df_ch4[sel_cols+["value"]]
 .drop_duplicates()
 .merge(
     df_ch4_eo[sel_cols+["value"]],
     on=sel_cols,
     how="outer",
     suffixes=("_gb", "_eo")
 )
)

# combine dataset with template data to get also all combinations
# where both, GB and EO are NaN
df_total = df_combined.merge(df_template, on=sel_cols, how="outer")

df_total["day"] = 1
df_total["date"] = pd.to_datetime(df_total[["year", "month", "day"]])
df_total = df_total.drop(columns=["year", "month", "day"]).set_index("date")
df_total.head()


# %%
def count_missings(df):
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
count_missings(df_total)

# %% [markdown]
# ## Predict GB from EO values
# ### Data preparation step

# %%
# collocated dataset (GB != NaN & EO != NaN)
df_collocated = (
    df_combined[(~df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)

# add variable indicating hemispheric level (Northern, Southern, Tropics)
conditions = [
    (df_collocated['lat'] > 45),
    (df_collocated['lat'] < -45)
]
choices = ['N', 'S']
df_collocated['hemisphere'] = np.select(conditions, choices, default='Tropics')

# add indicator variable dividing latitudinal direction into
# Northern and Southern hemisphere (no tropics)
df_collocated["X_N"] = np.where(df_collocated["lat"] > 0, 1, 0)
df_collocated.head()

# %% [markdown]
# ### Modelling step

# %%
# baseline model
m1 = bmb.Model("value_gb ~ 1 + value_eo", df_collocated)
# add trend and seasonal cycle
m2 = bmb.Model("value_gb ~ 1 + (1 | month) + year + value_eo", df_collocated, categorical=["month"])
# add offset varying by latitudes
m3 = bmb.Model("value_gb ~ 1 + (1 | month) + (1 | lat) + year + value_eo", df_collocated, categorical=["month", "lat"])
# add slope varying by hemisphere
m4 = bmb.Model("value_gb ~ 1 + (1 + X_N | month) + (1 | lat) + year + value_eo", df_collocated, categorical=["month", "lat"])

#iknots = [2,4,6,8,10]
m5 = bmb.Model("value_gb ~ 1 + year + value_eo + (1 + X_N | month) + (1|lat) + (1|lon)", df_collocated, categorical=["month","lat", "lon"])

# %%
m5 = bmb.Model("value_gb ~ 1 + year + value_eo + (1 + X_N | month) + (1|lat) + (1|lon)", df_collocated, categorical=["month","lat", "lon"])
idata5 = m5.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True), traget_accept=0.9)

# %% jupyter={"outputs_hidden": true}
idata1 = m1.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True))
idata2 = m2.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True))
idata3 = m3.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True))
idata4 = m4.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True))
idata5 = m5.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True))


# %%
def plot_model_comparison(idata_list):
    # compute Pareto-smoothed importance sampling leave-one-out cross validation
    for idata in idata_list:
        az.loo(idata)
    
    # compare models based on their expected log pointwise predictive density (ELPD)
    comp_df = az.compare({f"m{i+1}": idata for i, idata in enumerate(idata_list)})

    # plot model comparison
    axs = az.plot_compare(comp_df, figsize=(8,4))
    axs.tick_params(axis="x", labelsize=10)
    axs.set_title("Model comparison (higher is better)")
    
    return axs


# %% jupyter={"outputs_hidden": true}
plot_model_comparison([idata1, idata2, idata3, idata4, idata5]);


# %%
def plot_model_r2(idata_list, model_list, ylim):
    # compute R2
    r2_list = [model_list[i].r2_score(idata_list[i], summary=True) for i in range(len(idata_list))]
    
    # plot R2 for each model
    fig, axs = plt.subplots(figsize=(6,3))
    for i, r2_m in enumerate(r2_list):
        axs.plot([i], r2_m.r2, "o", color="black", zorder=2)
        axs.axhline(r2_m.r2, lw=0.5, color="grey", zorder=1)
    axs.set_xticks(range(len(r2_list)), [f"m{i+1}" for i in range(len(model_list))])
    axs.set_ylim(ylim)
    axs.set_xlabel("model specification")
    axs.set_ylabel(r"Bayesian $R^2$")
    axs.set_title("Prop. of variance in the data explained by the model")

    return fig, axs


# %%
plot_model_r2([idata1, idata2, idata3, idata4, idata5], [m1, m2, m3, m4, m5], ylim=(0.36, 0.45));


# %%
def plot_model_predictions(df_collocated, model, idata, gas, unit, fit=True, eo=False):
    if fit:
        model.predict(idata, kind="response")
        df_collocated["value_gb_pred"] = idata.posterior_predictive.value_gb.mean(dim=("chain", "draw")).values
    
    df_coll_agg = (
        df_collocated
            .reset_index(drop=True)
            .groupby(["year", "month", "hemisphere"])
            .agg({"value_gb": "mean", "value_eo": "mean", "value_gb_pred": "mean"})
            .reset_index()
    )
    
    df_coll_agg["day"] = 1
    df_coll_agg["date"] = pd.to_datetime(df_coll_agg[["year", "month", "day"]])
    
    fig, axs = plt.subplots(3,1, figsize=(7,7), constrained_layout=True, sharex=True)
    for i, hemi in enumerate(df_coll_agg.hemisphere.unique()):
        if i == 0:
            label_gb = "observed"
            label_pred = "predicted"
        else:
            label_gb = None
            label_pred = None
            
        sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb", 
                     lw=2, ax=axs[i], label=label_gb, alpha=0.3, color="grey")
        sns.scatterplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb", 
                     ax=axs[i], color="black", s=7)
        sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_gb_pred", 
                     lw=2, ax=axs[i], alpha=0.6, label=label_pred)
        if eo:
            sns.lineplot(data=df_coll_agg[df_coll_agg.hemisphere == hemi], x="date", y="value_eo", 
                     lw=2, ax=axs[i], alpha=0.6, label=label_pred)
        axs[i].set_title(f"hemisphere: {hemi}")
        axs[i].set_ylabel(f"{gas} in {unit}")
    axs[2].set_xlabel("year/month")
    axs[0].legend(frameon=False)

    return fig, axs


# %%
plot_model_predictions(df_collocated, model=m5, idata=idata5, gas="$CO_2$", unit="ppm");

# %%
m5.predict(idata5, kind="response")
df_collocated["value_gb_pred"] = idata5.posterior_predictive.value_gb.mean(dim=("chain", "draw")).values


# %% [markdown]
# ### Trend

# %%
def plot_predictions(df, lat=True, lon=True):
    df_year = df.groupby(["year"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
    df_month = df.groupby(["month"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
    
    fig, axs = plt.subplots(2,2, constrained_layout=True, figsize=(8,4))
    sns.lineplot(data=df_year, x="year", y="value_gb", label="observed", ax=axs[0,0])
    sns.lineplot(data=df_year, x="year", y="value_gb_pred", label="predicted", ax=axs[0,0], linestyle="--")
    axs[0,0].legend(frameon=False)
    
    sns.lineplot(data=df_month, x="month", y="value_gb", ax=axs[0,1])
    sns.lineplot(data=df_month, x="month", y="value_gb_pred", ax=axs[0,1], linestyle="--")

    if lat:
        df_lat = df.groupby(["lat"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
        sns.lineplot(data=df_lat, x="lat", y="value_gb", ax=axs[1,0])
        sns.lineplot(data=df_lat, x="lat", y="value_gb_pred", ax=axs[1,0], linestyle="--")
        
    if lon:
        df_lon = df.groupby(["lon"]).agg({"value_gb":"mean", "value_gb_pred":"mean"}).reset_index()
        sns.lineplot(data=df_lon, x="lon", y="value_gb", ax=axs[1,1])
        sns.lineplot(data=df_lon, x="lon", y="value_gb_pred", ax=axs[1,1], linestyle="--")

    return fig, axs


# %%
plot_predictions(df_collocated);

# %% [markdown]
# ## Predict new data given EO

# %%
#m5 = bmb.Model("value_gb ~ 1 + year + value_eo + (1 + X_N | month) + (1|lat) + (1|lon)", df_collocated, categorical=["month","lat", "lon"])
m5 = bmb.Model("value_gb ~ 1 + value_eo:year + (1 | month) + hsgp(lat, lon, by=X_N, c=1.5, m=10)", df_collocated, categorical=["month"])
idata5 = m5.fit(draws=1000, random_seed=SEED, idata_kwargs=dict(log_likelihood = True), target_accept=0.9)

# %%
df_pred_gb = (
    df_combined[(df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)

conditions = [
    (df_pred_gb['lat'] > 45),
    (df_pred_gb['lat'] < -45)
]
choices = ['N', 'S']
df_pred_gb['hemisphere'] = np.select(conditions, choices, default='Tropics')

df_pred_gb["time"] = df_pred_gb["year"] + (df_pred_gb["month"] - 1) / 12.0
df_pred_gb["time_sin"] = np.sin(2 * np.pi * df_pred_gb["time"])
df_pred_gb["time_cos"] = np.cos(2 * np.pi * df_pred_gb["time"])

df_pred_gb["X_N"] = np.where(df_pred_gb["lat"] > 0, 1, 0)
df_pred_gb.head()

# %%
df_gb_eo = pd.concat([df_collocated, df_pred_gb])

# %%
df_gb_eo = pd.concat([df_collocated, df_pred_gb]) #df_pred_gb

m5.predict(idata5, data=df_gb_eo, kind="response") #, sample_new_groups=True
df_gb_eo["value_gb_pred"] = idata5.posterior_predictive.value_gb.mean(dim=("chain", "draw")).values

# %%
plot_predictions(df_gb_eo);

# %% [markdown]
# ## Gaussian Process

# %% jupyter={"outputs_hidden": true}
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

long_term_trend_kernel = 40.0**2 * RBF(length_scale=20)

seasonal_kernel = (
    8.0**2
    * RBF(length_scale=0.1)
    * ExpSineSquared(length_scale=1.3, periodicity=1.0, periodicity_bounds="fixed")
)

irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

noise_kernel = 0.2**2 * RBF(length_scale=1.3) + WhiteKernel(
    noise_level=0.2**2, noise_level_bounds=(1e-5, 1e5)
)

value_eo_kernel = 1.0**2 * RBF(length_scale=1.0)

kernel = (
    long_term_trend_kernel + irregularities_kernel + noise_kernel + seasonal_kernel
)

df_collocated["time"] = df_fit_gb["year"] + (df_fit_gb["month"] - 1) / 12.0

X = df_collocated[["time", "value_eo", "lat", "lon"]].values
y = df_collocated.value_gb.to_numpy()
y_mean = df_collocated.value_gb.mean()

gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-9)
gaussian_process.fit(X, y-y_mean)

# %%
# CH4

df_collocated["time"] = df_collocated["year"] + (df_collocated["month"] - 1) / 12.0

X = df_collocated[["time", "lat", "lon", "value_eo"]].values
y = df_collocated.value_gb.to_numpy()
y_mean = df_collocated.value_gb.mean()

# Only sensitive to Time (col 0)
seasonal_kernel = (
    8.0**2
    * RBF(length_scale=0.1)
    * ExpSineSquared(length_scale=1.3, periodicity=1.0, periodicity_bounds="fixed")
)

# B. Long Term Trend
# Smooth trend over Time (0) and large regions of Lat/Lon (1,2)
long_term_trend_kernel = 40.0**2 * RBF(length_scale=[20.0, 50.0, 50.0, 1e5])


# C. EO Correction (The Regressor)
# This part relies mainly on the satellite data (col 3) to correct the prediction
value_eo_kernel = 1.0**2 * RBF(length_scale=[1e5, 1e5, 1e5, 2.0])


# D. Irregularities (Short term weather)
irregularities_kernel = 0.5**2 * RationalQuadratic(
    length_scale=1.0, alpha=1.0
)

# E. Noise
noise_kernel = 0.1**2 * RBF(length_scale=[0.1, 1e5, 1e5, 1e5]) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)


# 3. COMBINE AND FIT
# ------------------
kernel = (
    long_term_trend_kernel + 
    seasonal_kernel +       # Now lat-dependent!
    value_eo_kernel + 
    irregularities_kernel + 
    noise_kernel
)

gaussian_process = GaussianProcessRegressor(
    kernel=kernel, 
    normalize_y=False, 
    alpha=1e-5 # Increased for numerical stability
)

# Fit (Center Y for better performance if normalize_y=False)
gaussian_process.fit(X, y - y_mean)

# %% jupyter={"source_hidden": true}
# CO2
df_collocated["time"] = df_collocated["year"] + (df_collocated["month"] - 1) / 12.0

X = df_collocated[["time", "lat", "lon", "value_eo"]].values
y = df_collocated.value_gb.to_numpy()
y_mean = df_collocated.value_gb.mean()

# Only sensitive to Time (col 0)
seasonal_kernel = (
    8.0**2
    * RBF(length_scale=0.1)
    * ExpSineSquared(length_scale=1.3, periodicity=1.0, periodicity_bounds="fixed")
)

# B. Long Term Trend
# Smooth trend over Time (0) and large regions of Lat/Lon (1,2)
long_term_trend_kernel = 40.0**2 * RBF(length_scale=[20.0, 50.0, 50.0, 1e5])


# C. EO Correction (The Regressor)
# This part relies mainly on the satellite data (col 3) to correct the prediction
value_eo_kernel = 1.0**2 * RBF(length_scale=[1e5, 1e5, 1e5, 2.0])


# D. Irregularities (Short term weather)
irregularities_kernel = 0.5**2 * RationalQuadratic(
    length_scale=1.0, alpha=1.0
)

# E. Noise
noise_kernel = 0.1**2 * RBF(length_scale=[0.1, 1e5, 1e5, 1e5]) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)


# 3. COMBINE AND FIT
# ------------------
kernel = (
    long_term_trend_kernel + 
    seasonal_kernel +       # Now lat-dependent!
    value_eo_kernel + 
    irregularities_kernel + 
    noise_kernel
)

gaussian_process = GaussianProcessRegressor(
    kernel=kernel, 
    normalize_y=False, 
    alpha=1e-5 # Increased for numerical stability
)

# Fit (Center Y for better performance if normalize_y=False)
gaussian_process.fit(X, y - y_mean)

# %%
df_pred_gb = (
    df_combined[(df_combined.value_gb.isna()) & (~df_combined.value_eo.isna())]
        .drop_duplicates()
        .reset_index(drop=True)
)

conditions = [
    (df_pred_gb['lat'] > 45),
    (df_pred_gb['lat'] < -45)
]
choices = ['N', 'S']
df_pred_gb['hemisphere'] = np.select(conditions, choices, default='Tropics')

df_pred_gb["time"] = df_pred_gb["year"] + (df_pred_gb["month"] - 1) / 12.0

df_pred_gb["X_N"] = np.where(df_pred_gb["lat"] > 0, 1, 0)
df_pred_gb.shape

# %%
df_collocated.shape

# %%
df_gb_eo = pd.concat([df_collocated, df_pred_gb[df_pred_gb.year < 2010]])

X_test = df_gb_eo[["time", "lat", "lon", "value_eo"]].values
mean_y_pred = gaussian_process.predict(X_test, return_std=False)
df_gb_eo["value_gb_pred"] = mean_y_pred + y_mean

# %%
plot_predictions(df_gb_eo, lat=True, lon=True);

# %%
plot_model_predictions(df_gb_eo, model=None, idata=None, gas="$CO_2$", unit="ppm", fit=False, eo=True);

# %%
conditions = [
    (df2['lat'] > 45),
    (df2['lat'] < -45)
]
choices = ['N', 'S']
df2['hemisphere'] = np.select(conditions, choices, default='Tropics')
df2_agg = (
    df2
        .reset_index(drop=True)
        .groupby(["year", "month", "hemisphere"])
        .agg({"value_gb": "mean", "value_eo": "mean"})
        .reset_index()
)

df2_agg["day"] = 1
df2_agg["date"] = pd.to_datetime(df2_agg[["year", "month", "day"]])

fig, axs = plt.subplots(1,3, figsize=(10,3), constrained_layout=True)
for i, (hemi, name) in enumerate(zip(["S", "N","Tropics"], ["Southern Hemisphere", "Northern Hemisphere", "Tropics"])):
    sns.lineplot(data=df2_agg[df2_agg.hemisphere==hemi], x="date", y="value_gb", lw=2, ax=axs[i], label="GB")
    sns.lineplot(data=df2_agg[df2_agg.hemisphere==hemi], x="date", y="value_eo", lw=2, ax=axs[i], label="EO")
    
    axs[i].set_title(name);

# %% jupyter={"outputs_hidden": true}
df_coll["trend"] = idata.posterior.Intercept.mean().values + idata.posterior.year.mean().values * df_coll.year + idata.posterior.value_eo.mean().values * df_coll.value_eo
df_coll_month = df_coll.reset_index().groupby(["month"]).agg({"trend": "mean"}).reset_index() 
df_coll_month["season_offset"] = idata.posterior["1|month"].mean(dim=("chain", "draw")).values
df_coll_month["seasonality"] = df_coll_month["trend"] + df_coll_month["season_offset"]

#df_coll["seasonality"] = idata.posterior.Intercept.mean().values + idata.posterior.year.mean().values * df_coll.year + idata.posterior.month.mean().values + idata.posterior.value_eo.mean().values * df_coll.value_eo

df_trend = df_coll.groupby("year").agg({"trend":"mean"}).reset_index()
#plt.plot(df_trend.year, df_trend.trend)

#sns.lineplot(data=df_coll_month, x="month", y="seasonality")

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
