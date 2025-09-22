"""
bayesian time series regression to model ghg concentrations
"""

from itertools import product
from typing import Any, Optional

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore

tfd = tfp.distributions  # type: ignore
root = tfd.JointDistributionCoroutine.Root


def compute_X_source(  # noqa: PLR0913
    observed: bool,
    d: pd.Series,
    n_years_before: int,
    n_years_after: int,
    n_lats: int,
    n_months: int = 12,
) -> np.ndarray:
    """
    Compute predictor design variable for grouping variable "source"

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    d :
        observed values

    n_years_pred :
        number of predicted years

    n_months :
        number of month

    n_lats :
        number of latitudes

    Returns
    -------
    :
        design matrix for predictor
    """
    source = pd.get_dummies(d.source).values
    n_groups = len(d.source.unique())

    if len(np.unique(d.month)) != 12:  # noqa: PLR2004
        raise AssertionError(  # noqa: TRY003
            f"Expected 12 months per year but got {len(np.unique(d.month))}"
        )

    if observed:
        X_source = source
    else:
        n_pred_before = n_months * n_years_before * n_lats
        n_pred_after = n_months * n_years_after * n_lats
        # Create one-hot encodings for predicted data
        X_source_before = np.transpose(
            np.tile(np.eye(n_groups).astype(bool), n_pred_before)
        )
        X_source_after = np.transpose(
            np.tile(np.eye(n_groups).astype(bool), n_pred_after)
        )
        X_source = np.concat([X_source_before, source, X_source_after])

    return X_source


def compute_X_seasonality(  # noqa: PLR0913
    observed: bool,
    d: pd.DataFrame,
    n_years_before: int,
    n_years_after: int,
    n_months: int = 12,
    n_groups: int = 1,
) -> np.ndarray:
    """
    Compute predictor design variable for seasonality

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    d :
        data frame with observed values

    n_years_before :
        number of years to extrapolate

    n_years_after :
        number of years to interpolate

    n_months :
        number of month

    n_groups :
        number of groups (levels in grouping var)

    Returns
    -------
    :
        design matrix for predictor
    """
    # Create one-hot encodings for observed data
    if len(np.unique(d.month)) != 12:  # noqa: PLR2004
        raise AssertionError(  # noqa: TRY003
            f"observed number of months should be 12, but is {len(np.unique(d.month))}"
        )

    seasonality = np.eye(n_months)[np.array(d.month, dtype=int) - 1]

    if observed:
        X_seasonality = seasonality
        shapes = (0, seasonality.shape[0], 0)
    else:

        def compute_season_pred(n_years: int, n_months=n_months, n_groups=n_groups):
            n_total = n_years * n_months
            if n_total == 0:
                return np.empty((0, n_months))
            else:
                # Create one-hot encodings for predicted data
                # repeat every row according to groups
                return np.repeat(
                    np.eye(n_months)[np.arange(n_total) % n_months],
                    repeats=(len(d.lat.unique()) * n_groups),
                    axis=0,
                )

        X_seasonality = np.concatenate(
            [
                compute_season_pred(n_years_before),
                seasonality,
                compute_season_pred(n_years_after),
            ]
        )

        shapes = (
            compute_season_pred(n_years_before).shape[0],
            seasonality.shape[0],
            compute_season_pred(n_years_after).shape[0],
        )

    return X_seasonality, shapes


def compute_X_trend(
    observed: bool,
    d: pd.DataFrame,
    n_years_after: int,
    n_years_before: int,
    n_groups: int = 1,
) -> np.ndarray:
    """
    Compute predictor design variable for trend

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    d :
        pandas DataFrame with obs. values

    n_years_before :
        number of years to extrapolate

    n_years_after :
        number of years to interpolate

    n_groups :
        number of groups (levels in grouping var)

    Returns
    -------
    :
        design matrix for predictor
    """
    # Linear trend feature
    nobs = d.shape[0]

    if observed:
        X_trend = np.linspace(0.0, 1.0, nobs, dtype=np.float32)[:, None]
    else:
        n_pred = (
            len(d.month.unique())
            * len(d.lat.unique())
            * (n_years_before + n_years_after)
        )
        X_trend = np.linspace(0.0, 1.0, n_pred + nobs, dtype=np.float32)[:, None]

    return X_trend


def compute_X_latitude(
    observed: bool,
    d: pd.DataFrame,
    n_years_before: int,
    n_years_after: int,
    n_groups: int = 1,
) -> np.ndarray:
    """
    Compute predictor design variable for trend

    Parameters
    ----------
    observed:
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    d :
        dataframe with observed values

    n_years_before :
        number of years to extrapolate

    n_years_after :
        number of years to interpolated

    n_groups :
        number of groups (levels in grouping var)

    Returns
    -------
    :
        design matrix for predictor
    """
    lat_array = np.array(pd.get_dummies(d.lat), dtype=float)

    if observed:
        X_latitude = lat_array
    else:

        def compute_lat(n_years):
            lat_unique = np.array(pd.get_dummies(d.lat.unique()), dtype=float)
            if n_years == 0:
                return np.empty((0, len(d.lat.unique())))
            else:
                return np.repeat(
                    np.tile(lat_unique, (n_years * len(d.month.unique()), 1)),
                    repeats=n_groups,
                    axis=0,
                )

        X_latitude = np.concatenate(
            [compute_lat(n_years_before), lat_array, compute_lat(n_years_after)]
        )

    return X_latitude


def compute_X_time(
    observed: bool,
    d: pd.DataFrame,
    n_years_before: int,
    n_years_after,
    n_groups: int = 1,
) -> np.ndarray:
    """
    Compute time variable

    Parameters
    ----------
    observed :
        if true, the number of observed observations is
        used, else the total_number=observed+predicted
        is used

    d :
        dataframe with observed values

    n_years_before :
        number of years to extrapolate

    n_years_after :
        number of years to interpolate

    n_groups :
        number of groups (levels in grouping var)

    Returns
    -------
    :
        time variable
    """
    time_temp = np.array([f"{y}-{m:02d}" for y, m in zip(d.year.values, d.month)])

    if observed:
        X_time = time_temp
    else:

        def compute_time_pred(case: str, n_years: Optional[int]):
            if n_years == 0:
                return []
            else:
                months = np.repeat(
                    np.arange(1, len(d.month.unique()) + 1),
                    repeats=(len(d.lat.unique()) * n_groups),
                    axis=0,
                )
                if case == "extrapolate":
                    years = np.arange(d.year.values[0] - n_years, d.year.values[0])
                elif case == "observed":
                    years = np.arange(d.year.values[0], d.year.values[-1] + 1)
                elif case == "interpolate":
                    years = np.arange(
                        d.year.values[-1] + 1, d.year.values[-1] + 1 + n_years
                    )

                time_pred = np.array(
                    [f"{y}-{m:02d}" for y, m in product(years, months)]
                )
                return time_pred

        X_time = np.concatenate(
            [
                np.concatenate(
                    [
                        compute_time_pred("extrapolate", n_years=n_years_before),
                        time_temp,
                    ]
                ),
                compute_time_pred("interpolate", n_years=n_years_after),
            ]
        )

    return X_time


def compute_Xs(  # noqa: PLR0913
    d: pd.DataFrame,
    n_years_before: int,
    n_years_after: int,
    n_groups: int,
    observed: bool,
    incl_source: bool,
    incl_lat: bool,
) -> dict[str, np.ndarray]:
    """
    Get design matrices for predictor variables (wrapper function)

    Parameters
    ----------
    d:
        dataset with observed values

    n_years_before :
        number of years to extrapolate

    n_years_after :
        number of years to interpolate

    n_groups :
        number of groups in source variable
        (levels in source variable)

    observed :
        if true, design matrix for observed values
        (as in dataset) are constructed
        if false, design matrix for observed and
        predicted values (years other than obs.)
        are constructed

    incl_source :
        whether the variable "source" is included in
        the dataset

    incl_lat :
        whether the variable "lat" is included in
        the dataset

    Returns
    -------
    :
        dictionary of predictor design variables
    """
    X_seasonality, shapes = compute_X_seasonality(
        observed=observed,
        d=d,
        n_years_before=n_years_before,
        n_years_after=n_years_after,
        n_groups=n_groups,
    )

    X_trend = compute_X_trend(
        observed=observed,
        d=d,
        n_years_before=n_years_before,
        n_years_after=n_years_after,
        n_groups=n_groups,
    )
    X_time = compute_X_time(
        observed=observed,
        d=d,
        n_years_before=n_years_before,
        n_years_after=n_years_after,
        n_groups=n_groups,
    )

    res = dict(seasonality=X_seasonality, trend=X_trend, time=X_time)

    if incl_source:
        X_source = compute_X_source(
            observed=observed,
            d=d,
            n_years_before=n_years_before,
            n_years_after=n_years_after,
            n_months=12,
            n_lats=len(d.lat.unique()),
        )
        res["source"] = X_source

    if incl_lat:
        X_latitude = compute_X_latitude(
            observed=observed,
            d=d,
            n_years_before=n_years_before,
            n_years_after=n_years_after,
            n_groups=n_groups,
        )
        res["latitude"] = X_latitude

    return res, shapes


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


class PrepareGAM:
    """
    prepare hyperparameter for generalized additive model
    """

    def __init__(self, n_changepoints: int, X_seasonality: np.ndarray):
        self.n_changepoints = n_changepoints
        n_tp = X_seasonality.shape[0]
        self.t = np.linspace(0, 1, n_tp, dtype=np.float64)
        self.s = np.linspace(0, max(self.t), n_changepoints + 2, dtype=np.float64)[1:-1]
        self.A = (self.t[:, None] > self.s).astype(np.float64)
        # Generate seasonality design matrix
        # Set n=6 here so that there are 12 columns (same as `seasonality_all`)
        self.X_pred = self.gen_fourier_basis(
            np.where(X_seasonality)[1], p=X_seasonality.shape[-1], n=6
        )
        self.n_pred = self.X_pred.shape[-1]

    def __call__(self):
        """
        Return computed hyperparameter for GAM

        Returns
        -------
        :
            tuple with hyperparameters
        """
        return self.n_changepoints, self.t, self.s, self.A, self.X_pred, self.n_pred

    def gen_fourier_basis(self, t: np.ndarray, p: float, n: int) -> np.ndarray:
        """
        Compute Fourier basis functions

        Parameters
        ----------
        t :
            time step

        p :
            regular period of the time series (p=365.25 for yearly data or
            p=7 for weekly data)

        n :
            number of predictors divided by two

        Returns
        -------
        :
            results from Fourier basis functions
        """
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1, dtype=np.float64)


def run_gam_model(  # noqa: PLR0913
    d: pd.DataFrame,
    X_seasonality_pred: np.ndarray,
    X_latitude_pred: np.ndarray,
    X_source_pred: np.ndarray,
    prior: bool,
    shapes: tuple[int, ...],
) -> dict[str, Any]:
    """
    Run generalized additive model

    Parameters
    ----------
    d:
        dataset with observed data

    X_seasonality_pred:
        design matrix for "seasonality" variable
        incl. future (non-observed) years

    X_latitude_pred :
        design matrix for "latitude" variable
        incl. future (non-observed) years

    X_source_pred :
        design matrix for "source" variable
        incl. future (non-observed) years

    prior :
        whether prior samples should be drawn and
        prior predictions be computed
        if false; model is fitted and posterior
        predictions are returned

    shapes :
        information about indices where observed
        data in full-design matrix starts-stops

    Returns
    -------
    :
        results from model run either
        prior samples/prior predictions or
        fitted model/posterior predictions
    """
    prepare_gam = PrepareGAM(n_changepoints=12, X_seasonality=X_seasonality_pred)
    n_changepoints, t, s, A, X_pred, n_pred = prepare_gam()

    @tfd.JointDistributionCoroutine
    def gam():
        beta = yield root(
            tfd.Sample(tfd.Normal(0.0, 1.0), sample_shape=n_pred, name="beta")
        )
        seasonality_gam = tf.einsum("ij,...j->...i", X_pred, beta)

        k = yield root(tfd.HalfNormal(10.0, name="k"))
        m = yield root(
            tfd.Normal(tf.cast(d.value.mean(), tf.float32), scale=5.0, name="m")
        )
        tau = yield root(tfd.HalfNormal(10.0, name="tau"))
        delta = yield tfd.Sample(
            tfd.Laplace(0.0, tau), sample_shape=n_changepoints, name="delta"
        )
        source_sd = yield tfd.HalfCauchy(loc=0.0, scale=5.0, name="source_sd")

        varying_intercept = yield root(
            tfd.Sample(
                tfd.Normal(loc=0, scale=source_sd),
                sample_shape=(2),
                name="varying_intercept",
            )
        )
        latitude = yield root(
            tfd.Sample(
                tfd.Normal(loc=0.0, scale=10.0),
                sample_shape=X_latitude_pred.shape[-1],
                name="latitude",
            )
        )

        growth_rate = k[..., None] + tf.einsum("ij,...j->...i", A, delta)
        offset = m[..., None] + tf.einsum("ij,...j->...i", A, -s * delta)
        group = tf.einsum("ij,...j->...i", X_source_pred, varying_intercept)
        trend = growth_rate * t + offset + group
        lat = tf.einsum("ij,...j->...i", X_latitude_pred, latitude)

        y_hat = seasonality_gam + trend + lat
        y_hat = y_hat[..., shapes[0] : (shapes[0] + shapes[1])]

        noise_sigma = yield root(tfd.HalfNormal(scale=5.0, name="noise_sigma"))
        observed = yield tfd.Independent(  # noqa: F841
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )

    if prior:
        prior_samples = get_prior_samples(model=gam, n_samples=1000)
        prior_predictions = get_prior_predictions(prior_samples)

        return {"prior_pred": prior_predictions, "prior_samples": prior_samples}

    else:
        mcmc_samples, fitted_model = fit_model(model=gam, y_obs=d.value)

        seasonality = tf.einsum("ij,...j->...i", X_pred, mcmc_samples.beta)
        growth_rate = mcmc_samples.k[..., None] + tf.einsum(
            "ij,...j->...i", A, mcmc_samples.delta
        )
        offset = mcmc_samples.m[..., None] + tf.einsum(
            "ij,...j->...i", A, -s * mcmc_samples.delta
        )
        group = tf.einsum(
            "ij,...j->i...", X_source_pred, mcmc_samples.varying_intercept
        )
        trend = growth_rate * t + offset + tf.transpose(group, [1, 2, 0])
        latitude = tf.einsum("ij,...j->i...", X_latitude_pred, mcmc_samples.latitude)

        mu = seasonality + trend + tf.transpose(latitude, perm=[1, 2, 0])

        y_pred = tfd.Normal(mu, mcmc_samples.noise_sigma[..., None]).sample()
        posterior_pred = tf.transpose(y_pred, perm=[1, 0, 2])

        return {
            "posterior_pred": posterior_pred,
            "mcmc_samples": mcmc_samples,
            "fitted_model": fitted_model,
        }


def run_simple_reg_model(  # noqa PLR0913
    d: pd.DataFrame,
    X_seasonality: np.ndarray,
    X_trend: np.ndarray,
    X_latitude: np.ndarray,
    prior: bool,
    X_trend_pred: Optional[np.ndarray] = None,
    X_seasonality_pred: Optional[np.ndarray] = None,
    X_latitude_pred: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Run simple regression model

    Parameters
    ----------
    d:
        dataset with observed values

    X_seasonality:
        design matrix of seasonality
        only for observed years

    X_trend:
        design matrix of trend
        only for observed years

    X_latitude :
        design matrix of latitude
        only for observed yéars

    prior :
        whether prior samples/prior predictions
        should be computed
        if false the model is fitted and posterior
        predictions are returned

    X_trend_pred :
        design matrix of trend incl.
        non-observed years (forecast)

    X_seasonality_pred :
        design matrix of seasonality incl.
        non-observed years (forecast)

    X_latitude_pred :
        design matrix of latitude variable incl
        non-observed years (forecast)

    Returns
    -------
    :
        dictionary with results incl. prior
        samples/prior predictions or posterior
        predictions/fitted model
    """

    @tfd.JointDistributionCoroutine
    def ts_regression_model():
        intercept = yield root(
            tfd.Normal(
                loc=tf.cast(d.value.mean(), tf.float32), scale=10.0, name="intercept"
            )
        )

        trend = yield root(tfd.Normal(loc=0.0, scale=10.0, name="trend"))

        seasonality = yield root(
            tfd.Sample(
                tfd.Normal(loc=0.0, scale=1.0),
                sample_shape=X_seasonality.shape[-1],
                name="seasonality",
            )
        )

        latitude = yield root(
            tfd.Sample(
                tfd.Normal(loc=0.0, scale=10.0),
                sample_shape=X_latitude.shape[-1],
                name="latitude",
            )
        )

        random_noise = yield root(
            tfd.HalfCauchy(loc=0.0, scale=5.0, name="random_noise")
        )

        mu = (
            intercept[..., None]
            + tf.einsum("ij,...->...i", X_trend, trend)
            + tf.einsum("ij,...j->...i", X_seasonality, seasonality)
            + tf.einsum("ij,...j->...i", X_latitude, latitude)
        )

        yield tfd.Independent(
            tfd.Normal(mu, random_noise[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )

    if prior:
        prior_samples = get_prior_samples(model=ts_regression_model, n_samples=1000)
        prior_predictions = get_prior_predictions(prior_samples)
        return {"prior_pred": prior_predictions, "prior_samples": prior_samples}

    else:
        mcmc_samples, fitted_model = fit_model(model=ts_regression_model, y_obs=d.value)

        # posterior predictions
        trend = tf.einsum("ij,...->i...", X_trend_pred, mcmc_samples.trend)
        seasonality = tf.einsum(
            "ij,...j->i...", X_seasonality_pred, mcmc_samples.seasonality
        )
        latitude = tf.einsum("ij,...j->i...", X_latitude_pred, mcmc_samples.latitude)
        mu = mcmc_samples.intercept + trend + seasonality + latitude
        y_pred = tfd.Normal(mu, mcmc_samples.random_noise).sample()
        posterior_pred = tf.transpose(y_pred, [2, 1, 0])

        return {
            "posterior_pred": posterior_pred,
            "mcmc_samples": mcmc_samples,
            "fitted_model": fitted_model,
        }


def run_mlm_model(  # noqa: PLR0913
    d: pd.DataFrame,
    X_seasonality: np.ndarray,
    X_trend: np.ndarray,
    X_source: np.ndarray,
    prior: bool,
    X_seasonality_pred: Optional[np.ndarray] = None,
    X_trend_pred: Optional[np.ndarray] = None,
    X_source_pred: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Run hierarchical regression model

    Parameters
    ----------
    d:
        dataset with observed values

    X_seasonality:
        design matrix of seasonality
        only for observed years

    X_trend:
        design matrix of trend
        only for observed years

    X_latitude :
        design matrix of latitude
        only for observed yéars

    prior :
        whether prior samples/prior predictions
        should be computed
        if false the model is fitted and posterior
        predictions are returned

    X_trend_pred :
        design matrix of trend incl.
        non-observed years (forecast)

    X_seasonality_pred :
        design matrix of seasonality incl.
        non-observed years (forecast)

    X_latitude_pred :
        design matrix of latitude variable incl
        non-observed years (forecast)

    Returns
    -------
    :
        dictionary with results incl. prior
        samples/prior predictions or posterior
        predictions/fitted model
    """

    @tfd.JointDistributionCoroutine
    def ts_multilevel_model():
        intercept = yield root(
            tfd.Normal(
                loc=tf.cast(d.value.mean(), tf.float32), scale=10.0, name="intercept"
            )
        )

        source_sd = yield tfd.HalfCauchy(loc=0.0, scale=5.0, name="source_sd")

        varying_intercept = yield root(
            tfd.Sample(
                tfd.Normal(loc=0, scale=source_sd),
                sample_shape=(2),
                name="varying_intercept",
            )
        )

        trend = yield root(tfd.Normal(loc=0.0, scale=10.0, name="trend"))

        seasonality = yield root(
            tfd.Sample(
                tfd.Normal(loc=0.0, scale=1.0),
                sample_shape=X_seasonality.shape[-1],
                name="seasonality",
            )
        )

        random_noise = yield root(
            tfd.HalfCauchy(loc=0.0, scale=5.0, name="random_noise")
        )

        mu = (
            intercept[..., None]
            + tf.einsum("ij,...j->...i", X_source, varying_intercept)
            + tf.einsum("ij,...->...i", X_trend, trend)
            + tf.einsum("ij,...j->...i", X_seasonality, seasonality)
        )

        yield tfd.Independent(
            tfd.Normal(mu, random_noise[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )

    if prior:
        prior_samples = get_prior_samples(model=ts_multilevel_model, n_samples=1000)
        prior_predictions = get_prior_predictions(prior_samples)
        return {"prior_pred": prior_predictions, "prior_samples": prior_samples}

    else:
        mcmc_samples, fitted_model = fit_model(model=ts_multilevel_model, y_obs=d.value)

        # posterior predictions
        intercept = mcmc_samples.intercept[..., None] + tf.einsum(
            "ij,...j->i...", X_source_pred, mcmc_samples.varying_intercept
        )
        trend = tf.einsum("ij,...->i...", X_trend_pred, mcmc_samples.trend)
        seasonality = tf.einsum(
            "ij,...j->i...", X_seasonality_pred, mcmc_samples.seasonality
        )
        mu = intercept + trend + seasonality
        y_pred = tfd.Normal(mu, mcmc_samples.random_noise).sample()
        posterior_pred = tf.transpose(y_pred, [2, 1, 0])

        return {
            "posterior_pred": posterior_pred,
            "mcmc_samples": mcmc_samples,
            "fitted_model": fitted_model,
        }


def run_latent_ar_gam_model(
    d: pd.DataFrame,
    X_seasonality: np.ndarray,
    prior: bool,
) -> dict[str, Any]:
    """
    Run generalized additive model with latent AR

    Parameters
    ----------
    d:
        dataset with obverved values

    X_seasonality :
        design matrix of seasonality

    prior:
        whether to run prior-only model (getting
        back prior samples/prior predictions) or
        fitting model (getting back posteriors)

    Returns
    -------
    :
        restructured dictionary with results
    """
    prepare_gam = PrepareGAM(n_changepoints=12, X_seasonality=X_seasonality)
    n_changepoints, t, s, A, X_pred, n_pred = prepare_gam()

    def gam_trend_seasonality():
        beta = yield root(
            tfd.Sample(tfd.Normal(0.0, 1.0), sample_shape=n_pred, name="beta")
        )
        seasonality = tf.einsum("ij,...j->...i", X_pred, beta)

        k = yield root(tfd.HalfNormal(10.0, name="k"))
        m = yield root(
            tfd.Normal(tf.cast(d.value.mean(), tf.float32), scale=5.0, name="m")
        )
        tau = yield root(tfd.HalfNormal(10.0, name="tau"))
        delta = yield tfd.Sample(
            tfd.Laplace(0.0, tau), sample_shape=n_changepoints, name="delta"
        )

        growth_rate = k[..., None] + tf.einsum("ij,...j->...i", A, delta)
        offset = m[..., None] + tf.einsum("ij,...j->...i", A, -s * delta)
        trend = growth_rate * t + offset
        noise_sigma = yield root(tfd.HalfNormal(scale=5.0, name="noise_sigma"))
        return seasonality, trend, noise_sigma

    def generate_gam_ar_latent(training_set=True):
        @tfd.JointDistributionCoroutine
        def gam_with_latent_ar():
            seasonality, trend, noise_sigma = yield from gam_trend_seasonality()

            # Latent AR(1)
            ar_sigma = yield root(tfd.HalfNormal(0.1, name="ar_sigma"))
            rho = yield root(tfd.Uniform(-1.0, 1.0, name="rho"))

            def ar_fun(y):
                loc = (
                    tf.concat([tf.zeros_like(y[..., :1]), y[..., :-1]], axis=-1)
                    * rho[..., None]
                )
                return tfd.Independent(
                    tfd.Normal(loc=loc, scale=ar_sigma[..., None]),
                    reinterpreted_batch_ndims=1,
                )

            temporal_error = yield tfd.Autoregressive(
                distribution_fn=ar_fun,
                sample0=tf.zeros_like(trend),
                num_steps=trend.shape[-1],
                name="temporal_error",
            )

            # Linear prediction
            y_hat = seasonality + trend + temporal_error
            if training_set:
                y_hat = y_hat[..., : d.shape[0]]

            # Likelihood
            yield tfd.Independent(
                tfd.Normal(y_hat, noise_sigma[..., None]),
                reinterpreted_batch_ndims=1,
                name="observed",
            )

        return gam_with_latent_ar

    gam_with_latent_ar = generate_gam_ar_latent()

    if prior:
        prior_samples = get_prior_samples(model=gam_with_latent_ar, n_samples=1000)
        prior_predictions = get_prior_predictions(prior_samples)
        return {"prior_pred": prior_predictions, "prior_samples": prior_samples}

    else:
        mcmc_samples, fitted_model = fit_model(model=gam_with_latent_ar, y_obs=d.value)

        # posterior predictions
        seasonality = tf.einsum("ij,...j->...i", X_pred, mcmc_samples.beta)
        growth_rate = mcmc_samples.k[..., None] + tf.einsum(
            "ij,...j->...i", A, mcmc_samples.delta
        )
        offset = mcmc_samples.m[..., None] + tf.einsum(
            "ij,...j->...i", A, -s * mcmc_samples.delta
        )
        trend = growth_rate * t + offset

        mu = seasonality + trend + mcmc_samples.temporal_error

        y_pred = tfd.Normal(mu, mcmc_samples.noise_sigma[..., None]).sample()

        posterior_pred = tf.transpose(y_pred, perm=[1, 0, 2])

        return {
            "posterior_pred": posterior_pred,
            "mcmc_samples": mcmc_samples,
            "fitted_model": fitted_model,
        }


def combine_pred_obs(
    d: pd.DataFrame,
    posterior_predictions: Any,
    X_pred_time: np.ndarray,
    n_years_pred: int,
    n_months: int = 12,
) -> pd.DataFrame:
    """
    Combine observed values with fitted and predicted values

    Parameters
    ----------
    d:
        dataframe with observed values

    posterior_predictions :
        posterior predictions from fitted model

    n_years_pred :
        number of years predicted

    n_months :
        number of months considered

    Returns
    -------
    :
        dataframe with observed/fitted and predicted values
    """
    # compute dataframe with predictions
    df_pred = pd.DataFrame(
        {
            "time": pd.to_datetime(X_pred_time[d.shape[0] :]),
            "lat": np.tile(d.lat.unique()[:, None], (n_months * n_years_pred, 1))[:, 0],
            "ypred_mean": tf.reduce_mean(posterior_predictions[0, :, d.shape[0] :], 0),
            "ypred_sd": tf.math.reduce_std(
                posterior_predictions[0, :, d.shape[0] :], 0
            ),
        }
    )

    df_pred["year"] = df_pred["time"].dt.year
    df_pred["month"] = df_pred["time"].dt.month
    df_pred["value"] = 0

    # add fitted values to corresponding observed values
    d2 = d.copy()
    d2["ypred_mean"] = tf.reduce_mean(posterior_predictions[0, :, : d.shape[0]], 0)
    d2["ypred_sd"] = tf.math.reduce_std(posterior_predictions[0, :, : d.shape[0]], 0)

    # combine observed/fitted values and predictions
    df = pd.concat([d2, df_pred])
    # overwrite time to have less issues with seaborn due to datetime
    df["time"] = pd.to_datetime(
        pd.DataFrame({"year": df.year, "month": df.month, "day": 16})
    )
    # additional time obj which might sometimes be better for plotting
    df["year_month"] = df.time.astype(str)
    return df
