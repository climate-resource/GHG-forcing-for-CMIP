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
d_raw = pd.read_csv("../../data/downloads/co2/co2_joint_comparison.csv")

d_long = d_raw.melt(id_vars=["time", "year", "month", "lon", "lat"],
             value_vars=["satellite", "ground_based_AK"],
             var_name="source")

d = compute_weighted_avg(d_long, ["time", "year", "month", "lat"])
d = d[d.lat.isin([-57.5, -52.5, -47.5, -42.5, -37.5, -32.5, -27.5, -22.5, -17.5,
       -12.5,  -7.5,  -2.5,   2.5,   7.5,  12.5,  17.5,  22.5,  27.5,
        32.5,  37.5,  42.5,  47.5,  52.5,  57.5])].reset_index(drop=True)
d_group = compute_weighted_avg(d_long, ["time", "year", "month", "lat", "source"])

d.head()


# %%
np.sort(d_long.lat.unique())

# %% [markdown]
# ## Generalized Additive Model (GAM)
# ### Improve trend assumption
# Model trend as semi-smooth step linear function; allow slope to change at some specific break points
# $$
# g(t) = (\kappa+\mathbf A\delta)t + (m+ \mathbf A\gamma)
# $$
# with
#
# + $\kappa$ global growth rate
# + $\delta$ rate adjustments at each change point (drift effect)
# + $m$ global intercept
# + $\mathbf A$ $n_t \times n_s$ matrix accumulating the drift effect $\delta$ of the slope whereby $n_s$ are the change points
# + $\gamma$ smoothing component
#
# ### Improve seasonality assumption
# Model seasonality as B-spline or Fourier Basis Function
# Fourier basis functions are a collection of sine and cosine functions that can be used for
# approximating arbitrary smooth seasonal effects
# $$
# s(t) = \sum_{n=1}^N\left[a_n \cos\left(\frac{2\pi nt}{P}\right)+b_n \sin\left(\frac{2\pi nt}{P}\right) \right]
# $$
# with
#
# + $P$ being the regular period that the time series has (365.25 for yearly data)
# + Fitting the seasonality using a design matrix generated from Fourier basis function requires estimating $2N$ parameters $\beta = [\alpha_1, \beta_1, \ldots,\alpha_N,\beta_N]$
#
# ### Probabilistic model
# $$
# \begin{align*}
#     n &\leftarrow 6 &\text{number of seasons}\\
#     P &\leftarrow ? &\text{period of time series}\\
#     \beta &= [\alpha_1,\beta_1,\ldots,\alpha_N,\beta_N] \sim \text{Normal}(0.,1.) &\text{Fourier Basis param.}\\
#     \kappa &\sim \text{HalfNormal}(10.) & \text{global growth rate}\\
#     m &\sim \text{Normal}(\bar{y}, 5.) &\text{global intercept}\\
#     \tau &\sim \text{HalfNormal}(10.) &\text{hyppar. shrinkage}\\
#     \delta &\sim \text{Laplace}(0., \tau) &\text{rate adjust.}\\
#     \sigma &\sim \text{HalfNormal}(10.) &\text{random noise}\\
#     \beta &\sim \text{Normal}(0,10) & \text{latitude}\\
#     \\
#     \gamma &=(-s \cdot \delta) & \text{smoothing compo.}\\
#     \text{gr} &=\kappa + \mathbf A\cdot \delta & \text{growth-rate}\\
#     \text{os} &= m + \mathbf A \cdot \gamma & \text{offset}\\
#     \text{tr} &= \text{gr} \cdot t + \text{os} & \text{trend}\\
#     s &= \sum_{n=1}^N\left[a_n \cos\left(\frac{2\pi nt}{P}\right)+b_n \sin\left(\frac{2\pi nt}{P}\right) \right] &\text{seasonality}\\
#     \mu &= s + \text{tr} + \beta X^\text{lat} \\
#     y &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$

# %% [markdown]
# ### Design matrices

# %%
X_obs = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                   observed=True, incl_source=False)

X_obs["seasonality"].shape

# %% [markdown]
# ### Prior predictions

# %%
prior_res = bayesreg.run_gam_model(
    d, X_obs["seasonality"], X_obs["latitude"],True, None
)

# %%
plotting.plot_predictions(
    prior_res["prior_pred"], pd.to_datetime(X_obs["time"]),
    d.value, "co2", prior=True
)

# %% [markdown]
# ### Fitting & posterior predictions

# %%
X_preds = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                     observed=False, incl_source=False)

# %%
posterior_res = bayesreg.run_gam_model(
    d, X_preds["seasonality"], X_preds["latitude"], False, X_preds["latitude"]
)

# %%
plotting.plot_predictions(posterior_res["posterior_pred"][0,:,:],
                          pd.to_datetime(X_preds["time"]),
                          d.value, "ch4", prior=False)

# %%
az.summary(posterior_res["fitted_model"])

# %%
az.plot_trace(posterior_res["fitted_model"], compact=True);

# %%
df_combined = bayesreg.combine_pred_obs(
    d, posterior_res["posterior_pred"],
    X_preds["time"], n_years_pred=10
)
df_combined.head()

# %%
# Define bins (we add -90 and 90 to cover edge cases)
bins = [90, 60, 45, 30, 15,0, -15, -30, -45, -60, -90]

def bin_latitudes(df, bins, lat_value="latitude"):
    # pd.cut requires increasing order, so reverse bins and label accordingly
    bins = sorted(bins)  # ascending order

    # Assign bin labels
    return pd.cut(df[lat_value], bins=bins, labels=bins[1:])

df_combined["lat_bin"] = bin_latitudes(df_combined, bins=bins, lat_value="lat")
df_combined["ypred_var"] = df_combined.ypred_sd ** 2
df_combined.groupby(["time", "year", "month", "year_month", "lat_bin"]).agg(
    {"value":"mean", "ypred_mean":"mean", "ypred_var":"mean"}).reset_index()

# %%
plotting.plot_error(df_combined.head(d.shape[0]))

# %%
plotting.plot_seasonality(df, value="ypred_mean")

# %%
plotting.plot_annual(df, "ypred_mean")

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
df = df_combined.copy()
df["hemisphere"] = np.where(df.lat.values>0, "N", "S")
df["ypred_var"] = df.ypred_sd ** 2
df_hemi = df.groupby(["time","hemisphere","year"]).agg({"ypred_mean":"mean", "value":"mean", "ypred_var":"mean"}).reset_index()
df_hemi["ypred_sd"] = np.sqrt(df_hemi.ypred_var)
df_global = df.groupby(["time","year"]).agg({"ypred_mean":"mean", "value":"mean", "ypred_var":"mean"}).reset_index()
df_global["ypred_sd"] = np.sqrt(df_global.ypred_var)

sns.lineplot(data=df_hemi, x="time", y="ypred_mean", hue="hemisphere", lw=1)
for col, hem in zip(["blue", "orange"],["N","S"]):
    plt.fill_between(df_hemi[df_hemi.hemisphere==hem].time,
                     df_hemi[df_hemi.hemisphere==hem].ypred_mean-df_hemi[df_hemi.hemisphere==hem].ypred_sd,
                     df_hemi[df_hemi.hemisphere==hem].ypred_mean+df_hemi[df_hemi.hemisphere==hem].ypred_sd,
                     alpha=0.2, color=col)
sns.lineplot(data=df_global, x="time", y="ypred_mean", lw=1, label="global")
plt.fill_between(df_global.time, df_global.ypred_mean-df_global.ypred_sd,
                df_global.ypred_mean+df_global.ypred_sd, alpha=0.2, color="green")
for hem in ["S","N"]:
    sns.lineplot(data=df_hemi[(df_hemi.year<2022) & (df_hemi.hemisphere==hem)],
                 x="time", y="value", alpha=0.9, color="black",
                 lw=1, linestyle="-", legend=None)
sns.lineplot(data=df_global[df_global.year<2022], x="time", y="value",
             alpha=0.9, lw=1, linestyle="-", color="black")
plt.legend(frameon=False)

# %% [markdown]
# ## Autoregressive Models (AR)
#
# In an autoregressive model, the distribution of output at time $t$ is parameterized by a linear function of previous
# observations. Consider a first-order autoregressive model, referred to as AR(1), with a Gaussian likelihood
#
# $$
# y_t \sim \text{Normal}(\alpha + \rho\cdot y_{t-1}, \sigma)
# $$
#
# The location is a linear function of $y_{t-1}$.
#
# ### The model
# $$
# \begin{align*}
#     \rho &\sim \text{Uniform}(-1,1)\\
#     &\ldots\\
#     \mu &= s + \text{tr} \\
#     y &\sim \text{Normal}(\mu + \rho\cdot y_{t-1}, \sigma)
# \end{align*}
# $$
#
# ### Add latent AR component
# Instead of using an AR(k) likelihood, we can also include AR in a time series model by adding a latent AR component to the linear prediction.
#
# ### The model
# $$
# \begin{align*}
# \rho &\sim \text{Uniform}(-1,1)\\
# \sigma_{AR} &\sim \text{HalfNormal}(0.1)\\
# &\ldots\\
# \text{terr} &\sim \text{Normal}(\rho\cdot y_{t-1}, \sigma_{AR}) & \text{temporal error}\\
# \mu &= s + \text{tr} + \text{terr} \\
# y &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$
# Another way to interpret the explicit latent AR process is that it captures the temporally correlated residuals, so we expect the posterior estimation of the $\sigma_\text{noise}$ will be smaller compared to the model without this component.
#

# %%
X_obs = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                   observed=True, incl_source=False)

X_obs["seasonality"].shape

# %%
prior_res = bayesreg.run_latent_ar_gam_model(
    d, X_obs["seasonality"], prior=True
)

# %%
plotting.plot_predictions(
    prior_res["prior_pred"], pd.to_datetime(X_obs["time"]),
    d.value, "ch4", prior=True
)

# %%
X_preds = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                     observed=False, incl_source=False)

# %%
posterior_res = bayesreg.run_latent_ar_gam_model(
    d, X_preds["seasonality"], prior=False
)

# %%
plotting.plot_predictions(posterior_res["posterior_pred"][1,:,:],
                          pd.to_datetime(X_preds["time"]),
                          d.value, "ch4", prior=False)

# %%
az.summary(posterior_res["fitted_model"],
           var_names=['beta', 'tau', 'ar_sigma', 'rho', 'noise_sigma'],
          )

# %%
az.plot_trace(posterior_res["fitted_model"],
              var_names=['beta', 'tau', 'ar_sigma', 'rho', 'noise_sigma'],
              compact=True);
