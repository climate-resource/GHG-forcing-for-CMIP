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
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
root = tfd.JointDistributionCoroutine.Root

from ghg_forcing_for_cmip.data_comparison.utils import load_data
from ghg_forcing_for_cmip.data_assimilation import plotting
from ghg_forcing_for_cmip.data_assimilation import bayesian_regression as bayesreg

# %% [markdown]
# # Bayesian Time series regression

# %%
d_raw = load_data("ch4_binned.csv")
d = d_raw[d_raw.year < 2025].groupby(["time", "year","month"]).agg({"value":"mean"}).reset_index()
d.head()

# %% [markdown]
# ## Design matrix
# ### Seasonality

# %%
X_seasonality = bayesreg.compute_X_seasonality(
    observed=True, n_years_obs=len(d.year.unique()),
    n_years_pred=10
)
X_seasonality_pred = bayesreg.compute_X_seasonality(
    observed=False, n_years_obs=len(d.year.unique()),
    n_years_pred=10
)
X_seasonality

# %% [markdown]
# ### Trend

# %%
X_trend = bayesreg.compute_X_trend(
    observed=True, n_years_obs=len(d.year.unique()),
    n_years_pred=10
)
X_trend_pred = bayesreg.compute_X_trend(
    observed=False, n_years_obs=len(d.year.unique()),
    n_years_pred=10
)
X_trend[:5]

# %% [markdown]
# ### Time (year/month)

# %%
X_time = bayesreg.compute_X_time(
    observed=True, n_years_obs=len(d.year.unique()),
    n_years_pred=10, year_min=1983
)
X_time_pred = bayesreg.compute_X_time(
    observed=False, n_years_obs=len(d.year.unique()),
    n_years_pred=10, year_min=1983
)
X_time[:5]


# %% [markdown]
# ## Classical time series regression
# ### Probabilistic model
# $$
# \begin{align*}
#    \beta_0 &\sim \text{Normal}(\bar{y}_\text{obs},10) & \text{intercept}\\
#     \beta_1 &\sim \text{Normal}(0,10) & \text{trend}\\
#     \beta_2 &\sim \text{Normal}(0,1) & \text{seasonality}\\
#     \sigma &\sim \text{HalfCauchy}(0,5) & \text{random noise}\\
#     \\
#     \mu &= \beta_0+\beta_1X_1^\text{trend}+\beta_2X_2^\text{season.}\\
#     y_\text{obs} &\sim \text{Normal}(\mu, \sigma)
# \end{align*}
# $$

# %%
@tfd.JointDistributionCoroutine
def ts_regression_model():
    intercept = yield root(tfd.Normal(
        loc=tf.cast(d.value.mean(), tf.float32),
        scale=10., name="intercept"))

    trend = yield root(tfd.Normal(loc=0., scale=10., name="trend"))

    seasonality = yield root(tfd.Sample(
        tfd.Normal(loc=0., scale=1.),
        sample_shape=X_seasonality.shape[-1],
        name="seasonality"))

    random_noise = yield root(tfd.HalfCauchy(loc=0., scale=5., name="random_noise"))

    mu = (
        intercept[..., None] +
        tf.einsum("ij,...->...i", X_trend, trend) +
        tf.einsum("ij,...j->...i", X_seasonality, seasonality)
    )

    yield tfd.Independent(
        tfd.Normal(mu, random_noise[..., None]),
        reinterpreted_batch_ndims=1,
        name="observed"
    )


# %% [markdown]
# ### Prior Predictions

# %%
prior_samples = bayesreg.get_prior_samples(model=ts_regression_model, n_samples=1000)
prior_predictions = bayesreg.get_prior_predictions(prior_samples)

# %%
plotting.plot_predictions(prior_predictions, X_time, d.value, "ch4", prior=True)

# %% [markdown]
# ### Fit model and show posterior predictions

# %%
mcmc_samples, fitted_model = bayesreg.fit_model(model=ts_regression_model, y_obs=d.value)
posterior_predictions = bayesreg.get_posterior_predictions(mcmc_samples, X_trend_pred, X_seasonality_pred)

# %%
fitted_model

# %%
plotting.plot_predictions(posterior_predictions[0,:,:], X_time_pred, d.value, "ch4", prior=False)
