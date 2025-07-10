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

# %% [markdown]
# The methodological and computational background is heavily based on the *Time Series* chapter from the book [**Bayesian Modeling and Computation in Python**](https://bayesiancomputationbook.com/notebooks/chp_06.html) by Osvaldo, Ravin, and Junpeng
#
# Reference: Martin Osvaldo A, Kumar Ravin; Lao Junpeng. *Bayesian Modeling and Computation in Python* Boca Ratón, 2021. ISBN 978-0-367-89436-8

# %%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import arviz as az

tfd = tfp.distributions
root = tfd.JointDistributionCoroutine.Root

from ghg_forcing_for_cmip.data_comparison.utils import load_data, compute_weighted_avg
from ghg_forcing_for_cmip.data_assimilation import plotting
from ghg_forcing_for_cmip.data_assimilation import bayesian_regression as bayesreg

# %% [markdown]
# # Bayesian Time series regression
# ## Load data

# %%
d_raw = pd.read_csv("../../data/downloads/ch4/ch4_joint_comparison.csv")

d_long = d_raw.melt(id_vars=["time", "year", "month", "lon", "lat"],
             value_vars=["satellite", "ground_based_AK"],
             var_name="source")

d = compute_weighted_avg(d_long, ["time", "year", "month", "lat"])
d_group = compute_weighted_avg(d_long, ["time", "year", "month", "lat", "source"])

d.head()

# %%
d_group.head()

# %% [markdown]
# ## Design matrix for predictor variables

# %%
X_obs = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                   observed=True, incl_source=False)

# %% [markdown]
# ### Seasonality

# %%
X_obs["seasonality"].shape

# %% [markdown]
# ### Trend

# %%
X_obs["trend"].shape

# %% [markdown]
# ### Time (year/month)

# %%
X_obs["time"].shape

# %% [markdown]
# ### Latitude

# %%
X_obs["latitude"].shape

# %% [markdown]
# ## Classical time series regression
# ### Probabilistic model
# $$
# \begin{align*}
#    \beta_0 &\sim \text{Normal}(\bar{y}_\text{obs},10) & \text{intercept}\\
#     \beta_1 &\sim \text{Normal}(0,10) & \text{trend}\\
#     \beta_2 &\sim \text{Normal}(0,1) & \text{seasonality}\\
#     \beta_3 &\sim \text{Normal}(0,10) & \text{latitude}\\
#     \sigma &\sim \text{HalfCauchy}(0,5) & \text{random noise}\\
#     \\
#     \mu &= \beta_0+\beta_1X_1^\text{trend}+\beta_2X_2^\text{season.} + \beta_3X_3^\text{lat}\\
#     y_\text{obs} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$

# %% [markdown]
# ### Prior Predictions

# %%
prior_res = bayesreg.run_simple_reg_model(
    d, X_obs["seasonality"], X_obs["trend"], X_obs["latitude"],
    prior=True
)

# %%
plotting.plot_predictions(
    prior_res["prior_pred"], pd.to_datetime(X_obs["time"]),
    d.value, "ch4", prior=True
)

# %% [markdown]
# ### Fit model and show posterior predictions

# %%
X_pred = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                   observed=False, incl_source=False)
X_pred["latitude"].shape

# %%
posterior_res = bayesreg.run_simple_reg_model(
    d, X_obs["seasonality"], X_obs["trend"], X_obs["latitude"],
    prior=False,
    X_seasonality_pred=X_pred["seasonality"],
    X_trend_pred=X_pred["trend"],
    X_latitude_pred=X_pred["latitude"]
)

# %%
df_combined = bayesreg.combine_pred_obs(
    d, posterior_res["posterior_pred"], n_years_pred=10
)
df_combined.head()

# %%
plotting.plot_seasonality(df, value="ypred_mean")

# %%
plotting.plot_seasonality(df.head(d.shape[0]), value="value")

# %%
plotting.plot_annual(df, "ypred_mean")

# %%
plotting.plot_annual(df.head(d.shape[0]), "value")

# %%
plotting.plot_error(df)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

df["hemisphere"] = np.where(df.lat.values>0, "N", "S")
df["ypred_var"] = df.ypred_sd ** 2
df_hemi = df.groupby(["time","hemisphere","year"]).agg({"ypred_mean":"mean", "value":"mean", "ypred_var":"mean"}).reset_index()
df_hemi["ypred_sd"] = np.sqrt(df_hemi.ypred_var)
df_global = df.groupby(["time","year"]).agg({"ypred_mean":"mean", "value":"mean", "ypred_var":"mean"}).reset_index()

sns.lineplot(data=df_hemi, x="time", y="ypred_mean", hue="hemisphere", lw=1)
sns.lineplot(data=df_global, x="time", y="ypred_mean", lw=1)
sns.lineplot(data=df_hemi[df_hemi.year<2022], x="time", y="value", hue="hemisphere",
             alpha=0.6, lw=1, linestyle="--")
sns.lineplot(data=df_global[df_global.year<2022], x="time", y="value",
             label="global", alpha=0.6, lw=1, linestyle="--")

# %%
plotting.plot_predictions(posterior_res["posterior_pred"][0,:,:],
                          pd.to_datetime(X_pred["time"]),
                          d.value, "ch4", prior=False)

# %%
az.summary(posterior_res["fitted_model"])

# %%
az.plot_trace(posterior_res["fitted_model"], compact=True);

# %% [markdown]
# ## Include varying intercept for data source
# ### Probabilistic model
# $$
# \begin{align*}
#    \beta_0 &\sim \text{Normal}(\bar{y}_\text{obs},10) & \text{intercept}\\
#     \beta_1 &\sim \text{Normal}(0,10) & \text{trend}\\
#     \beta_2 &\sim \text{Normal}(0,1) & \text{seasonality}\\
#     \beta_3 &\sim \text{Normal}(0,10) & \text{latitude}\\
#     \sigma &\sim \text{HalfCauchy}(0,5) & \text{random noise}\\
#     \sigma_\text{source} &\sim \text{HalfCauchy}(0,5) & \text{sd source}\\
#     \\
#     u_{[i]} &\sim \text{Normal}(0,\sigma_\text{source}) & \text{varying intercept}\\
#     \mu_i &= \beta_0+ u_{[i]} + \beta_1X_1^\text{trend}+\beta_2X_2^\text{season.}+\beta_3X_3^\text{lat}\\
#     y_i^\text{obs} &\sim \text{Normal}(\mu_i, \sigma)
# \end{align*}
# $$

# %% [markdown]
# ### Design matrices

# %%
X_obs = bayesreg.compute_Xs(
    d_group, n_years_pred=10, n_groups=2,
    observed=True, incl_source=True
)

X_obs["seasonality"].shape

# %% [markdown]
# ### Prior predictions

# %%
prior_res = bayesreg.run_mlm_model(
    d_group, X_obs["seasonality"], X_obs["trend"], X_obs["source"],
    prior=True
)

# %%
plotting.plot_predictions_source(
    prior_res["prior_pred"],
    pd.to_datetime(X_obs["time"][::2]),
    d_group.value, "ch4", prior=True
)

# %% [markdown]
# ### Fitting & Posterior predictions

# %%
X_pred = bayesreg.compute_Xs(
    d_group, n_years_pred=10, n_groups=2,
    observed=False, incl_source=True
)

X_pred["seasonality"].shape

# %%
posterior_res = bayesreg.run_mlm_model(
    d_group,
    X_obs["seasonality"], X_obs["trend"], X_obs["source"],
    prior=False,
    X_seasonality_pred=X_pred["seasonality"],
    X_trend_pred=X_pred["trend"],
    X_source_pred=X_pred["source"],
)

# %%
plotting.plot_posterior_source(
    posterior_res["posterior_pred"][0,:,:],
    pd.to_datetime(X_pred["time"]),
    d_group.value, "ch4"
)

# %%
az.summary(posterior_res["fitted_model"])

# %%
az.plot_trace(posterior_res["fitted_model"], compact=True);
