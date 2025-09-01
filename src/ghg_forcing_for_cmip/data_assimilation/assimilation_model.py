"""
run Bayesian regression
"""

import arviz as az
import dill
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import xarray as xr

from ghg_forcing_for_cmip.data_assimilation import bayesian_regression as bayesreg
from ghg_forcing_for_cmip.data_assimilation import plotting
from ghg_forcing_for_cmip.data_comparison.utils import (
    compute_weighted_avg,
)

tfd = tfp.distributions
root = tfd.JointDistributionCoroutine.Root

gas = "co2"
n_years_after = 10
n_years_before = 10

d_joint = (
    xr.open_dataset(f"data/downloads/{gas}/{gas}_combined_q0.5.nc")
    .to_dataframe()
    .reset_index()
)

d_avg = (
    d_joint.groupby(["year", "month", "lon", "lat"])
    .agg({f"x{gas}": "mean", "value_vertical": "mean"})
    .reset_index()
)

d_long = d_avg.melt(
    id_vars=["year", "month", "lon", "lat"],
    value_vars=[f"x{gas}", "value_vertical"],
    var_name="source",
)

d_long.dropna(subset="value", inplace=True)

d_long["hemisphere"] = np.where(
    d_long.lat > 30.0,  # noqa: PLR2004
    "N",
    np.where(d_long.lat < -30.0, "S", "Tropics"),  # noqa: PLR2004
)
d = compute_weighted_avg(d_long, ["year", "month", "source", "hemisphere"])
d.rename(columns={"hemisphere": "lat"}, inplace=True)
d["time"] = d.year + d.month / 12

d = d[d.year.isin(np.arange(2003, 2023))].reset_index(drop=True)

# design matrices
X_obs, shapes_prior = bayesreg.compute_Xs(
    d,
    n_years_before=n_years_before,
    n_years_after=n_years_after,
    n_groups=2,
    observed=True,
    incl_source=True,
    incl_lat=True,
)

prior_res = bayesreg.run_gam_model(
    d,
    X_obs["seasonality"],
    X_obs["latitude"],
    X_obs["source"],
    True,
    shapes=shapes_prior,
)

plotting.plot_predictions(
    predictions=prior_res["prior_pred"],
    X_time_obs=pd.to_datetime(X_obs["time"]),
    X_time_pred=pd.to_datetime(X_obs["time"]),
    y_obs=d.value,
    d_group_obs=d.source,
    d_group_pred=d.source,
    ghg=gas,
    prior="prior",
    save_fig=True,
)

X_preds, shapes = bayesreg.compute_Xs(
    d,
    n_years_before=n_years_before,
    n_years_after=n_years_after,
    n_groups=2,
    observed=False,
    incl_source=True,
    incl_lat=True,
)

posterior_res = bayesreg.run_gam_model(
    d,
    X_preds["seasonality"],
    X_preds["latitude"],
    X_preds["source"],
    False,
    shapes=shapes,
)

plotting.plot_predictions(
    posterior_res["posterior_pred"][1, :, :],
    X_time_obs=pd.to_datetime(X_obs["time"]),
    X_time_pred=pd.to_datetime(X_preds["time"]),
    y_obs=d.value,
    d_group_obs=d.source,
    d_group_pred=np.where(X_preds["source"][:, 0], "value_vertical", f"x{gas}"),
    ghg=gas,
    prior="posterior",
    save_fig=True,
)

# get model summaries
az.summary(posterior_res["fitted_model"])
az.plot_trace(posterior_res["fitted_model"])

# save object
with open(f"report/results/{gas}_posterior_predictions.pkl", "wb") as f:
    dill.dump(posterior_res["posterior_pred"], f)

with open(f"report/results/{gas}_posterior_res.pkl", "wb") as f:
    dill.dump(posterior_res["fitted_model"], f)

d.to_csv(f"report/results/{gas}_obs_df.csv")
