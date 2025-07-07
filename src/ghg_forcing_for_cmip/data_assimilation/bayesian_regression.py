"""
bayesian time series regression to model ghg concentrations
"""

from itertools import product
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

tfd = tfp.distributions  # type: ignore
root = tfd.JointDistributionCoroutine.Root


def compute_X_seasonality(
    observed: bool, n_years_obs: int, n_years_pred: int, n_months: int = 12
) -> np.ndarray:
    """
    Compute predictor design variable for seasonality

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    n_years_obs :
        number of observed years

    n_years_pred :
        number of predicted years

    n_months :
        number of month

    Returns
    -------
    :
        design matrix for predictor
    """
    n_years = n_years_obs + n_years_pred

    # One-hot encode month indices for seasonality
    seasonality = np.eye(n_months, dtype=np.float32)[
        np.tile(np.arange(n_months), n_years)
    ]

    if not observed:
        X_seasonality = seasonality
    else:
        n_total = n_years_pred * n_months
        X_seasonality = seasonality[:-n_total, :]
    return X_seasonality


def compute_X_trend(
    observed: bool, n_years_obs: int, n_years_pred: int, n_months: int = 12
) -> np.ndarray:
    """
    Compute predictor design variable for trend

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    n_years_obs :
        number of observed years

    n_years_pred :
        number of predicted years

    n_months :
        number of month

    Returns
    -------
    :
        design matrix for predictor
    """
    n_years = n_years_obs + n_years_pred
    n_total = n_months * n_years

    # Linear trend feature
    trend = np.linspace(0.0, 1.0, n_total, dtype=np.float32)[:, None]

    if not observed:
        X_trend = trend
    else:
        n_total = n_years_pred * n_months
        X_trend = trend[:-n_total, :]
    return X_trend


def compute_X_time(
    observed: bool,
    n_years_obs: int,
    n_years_pred: int,
    year_min: int,
    n_months: int = 12,
) -> np.ndarray:
    """
    Compute time variable

    Parameters
    ----------
    observed :
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    n_years_obs :
        number of observed years

    n_years_pred :
        number of predicted years

    year_min:
        the minimum year

    n_months :
        number of month

    Returns
    -------
    :
        time variable
    """
    n_years = n_years_obs + n_years_pred

    # Monthly datetime labels
    months = np.arange(1, n_months + 1)
    years = np.arange(year_min, year_min + n_years)
    time = np.array(
        [f"{y}-{m:02d}" for y, m in product(years, months)], dtype="datetime64"
    )[:, None]

    if not observed:
        X_time = time
    else:
        n_total = n_years_pred * n_months
        X_time = time[:-n_total, :]
    return X_time


def get_prior_samples(
    model: tfd.JointDistributionCoroutine,  # type: ignore
    n_samples: int = 100,
) -> Any:
    """
    Get prior samples from probabilistic model

    Parameters
    ----------
    model :
        probabilistic time series model

    n_samples :
        number of prior samples

    Returns
    -------
    :
        Structured tuple including prior samples
    """
    prior_samples = model.sample(n_samples)
    return prior_samples


def get_prior_predictions(prior_samples: Any) -> tf.Tensor:
    """
    Get prior predictions of timeseries

    Parameters
    ----------
    prior_samples :
        Structured tuple including prior samples

    Returns
    -------
    :
        tf.Tensor with prior predictions
    """
    prior_predictions = prior_samples.observed
    return prior_predictions


def fit_model(
    model: tfd.JointDistributionCoroutine,  # type: ignore
    y_obs: pd.Series,
    n_chains: int = 4,
) -> tuple[Any, az.InferenceData]:
    """
    Run MCMC sampling to fit the probabilistic time series model to data

    Parameters
    ----------
    model :
        probabilistic time series model

    y_obs :
        observed time series

    n_chains :
        number of chains

    Returns
    -------
    :
        (mcmc_samples, fitted model)
    """
    # Wrap NUTS sampler with tf.function for compilation
    run_mcmc = tf.function(
        tfp.experimental.mcmc.windowed_adaptive_nuts, autograph=False, jit_compile=True
    )

    # Run MCMC sampling
    mcmc_samples, sampler_stats = run_mcmc(
        1000,
        model,
        n_chains=n_chains,
        num_adaptation_steps=1000,
        observed=tf.cast(np.array(y_obs)[None, ...], tf.float32),
    )

    # Convert samples to ArviZ InferenceData
    regression_idata = az.from_dict(
        posterior={
            k: np.swapaxes(v.numpy(), 1, 0) for k, v in mcmc_samples._asdict().items()
        },
        sample_stats={
            k: np.swapaxes(sampler_stats[k], 1, 0)
            for k in ["target_log_prob", "diverging", "accept_ratio", "n_steps"]
        },
    )

    return mcmc_samples, regression_idata


def get_posterior_predictions(
    mcmc_samples: Any, X_trend_pred: tf.Tensor, X_seasonality_pred: tf.Tensor
) -> tf.Tensor:
    """
    Compute posterior predictions from mcmc samples

    Parameters
    ----------
    mcmc_samples :
        mcmc samples from fitting stage

    X_trend_pred :
        trend design variable incl. predicted years

    X_seasonality_pred :
        seasonality design variable incl. predicted years

    Returns
    -------
    :
        posterior predictions (n_chains, samples, time_obs)
    """
    trend = mcmc_samples.intercept + tf.einsum(
        "ij,...->i...", X_trend_pred, mcmc_samples.trend
    )
    seasonality = tf.einsum(
        "ij,...j->i...", X_seasonality_pred, mcmc_samples.seasonality
    )
    mu = trend + seasonality
    y_pred = tfd.Normal(mu, mcmc_samples.random_noise).sample()
    return tf.transpose(y_pred, [2, 1, 0])
