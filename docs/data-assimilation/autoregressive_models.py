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

from ghg_forcing_for_cmip.data_comparison.utils import load_data
from ghg_forcing_for_cmip.data_assimilation import plotting
from ghg_forcing_for_cmip.data_assimilation import bayesian_regression as bayesreg

# %% [markdown]
# # Bayesian Time series regression
# ## Load data

# %%
d_raw = pd.read_csv("../../data/downloads/ch4/ch4_joint_long.csv")
d_filtered = d_raw.dropna(subset="value")[
    d_raw.source.isin(["satellite","ground_based_AK"])
].reset_index()


# %%
d = d_filtered.groupby(["time", "year","month"]).agg({"value":"mean"}).reset_index()
d_group = d_filtered.groupby(["time", "year","month","source"]).agg({"value":"mean"}).reset_index()

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

# %% [markdown]
# ### Design matrices

# %%
X_obs = bayesreg.compute_Xs(d, n_years_pred=10, n_groups=1,
                   observed=True, incl_source=False)

X_obs["seasonality"].shape

# %% [markdown]
# ### Prior predictions

# %%
prior_res = bayesreg.run_latent_ar_gam_model(
    d, X_obs["seasonality"], prior=True
)

# %%
plotting.plot_predictions(
    prior_res["prior_pred"], pd.to_datetime(X_obs["time"]),
    d.value, "ch4", prior=True
)

# %% [markdown]
# ### Fitting & posterior predictions

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
