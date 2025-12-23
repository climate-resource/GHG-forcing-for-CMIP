"""
analysis pipeline for incorporating EO into GB
"""

import itertools
from typing import Any, Optional

import arviz as az
import bambi as bmb
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import linear_model
from xgboost import XGBRegressor

from ghg_forcing_for_cmip import preprocessing


def fit_gb_from_eo(  # noqa: PLR0913
    formula: str,
    df: pd.DataFrame,
    seed: int,
    categorical: list[str],
    draws: int = 1000,
    chains: int = 4,
    cores: Optional[int] = None,
) -> Any:
    """
    Fit Bayesian regression model to predict GB from EO

    Parameters
    ----------
    formula :
        statistical model formulation

    df :
        data used for fitting

    seed :
        seed for reproducibility

    categorical :
        list with variables in formula considered
        to be categorical

    draws :
        draws for MCMC sampling

    chains :
        chains used for MCMC sampling

    cores :
        cores used for MCMC sampling;
        If `None`, it is equal to the number of CPUs in the system

    Returns
    -------
    Any
        model and inference object
    """
    model = bmb.Model(formula=formula, data=df, categorical=categorical)
    idata = model.fit(
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=seed,
        target_accept=0.95,
        inference_method="numpyro",
    )

    return model, idata


def check_rhat(idata: Any, critical_rhat: float = 1.01) -> pd.DataFrame:
    """
    Inspect Rhat diagnostic as one diagnostic for MCMC

    Parameters
    ----------
    idata :
        fitted inference object

    critical_rhat :
        rhat threshold; values above this threshold
        are considered as problematic

    Returns
    -------
    :
        number of variables with Rhat > critical_Rhat and
        details about corresponding model parameters
    """
    res = az.summary(idata)
    rhat_critical = res[res.r_hat > critical_rhat]

    print(f"Number of Rhat > {critical_rhat}: {len(rhat_critical)}.")

    return rhat_critical


def predict_gb(
    idata: Any,
    model: Any,
    in_sample_predictions: bool,
    test_data: Optional[pd.DataFrame] = None,
    dv_name: str = "value_gb",
) -> npt.NDArray:
    """
    Posterior predictions of dependent variable

    Parameters
    ----------
    idata :
        inference data object

    model :
        fitted model

    in_sample_predictions :
        whether predictions are based on data passed during
        fitting or new data is used

    test_data :
        if out-of-sample predictions, this argument
        specifies the external data

    dv_name :
        name of the predictor variable in the model

    Returns
    -------
    :
        predicted response variable
    """
    if (not in_sample_predictions) & (test_data is None):
        raise ValueError("If in_sample_predictions=False, test_data must be specified.")  # noqa: TRY003

    if in_sample_predictions:
        model.predict(idata, kind="response")
    else:
        model.predict(idata, data=test_data, kind="response", sample_new_groups=True)

    return idata.posterior_predictive[dv_name].mean(dim=("chain", "draw")).values


def do_gap_filling(
    df_partial_coverage: pd.DataFrame, df_missing: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict missing data (gap filling) to get GHG measures that show full coverage

    Parameters
    ----------
    df_partial_coverage :
        collocated and predicted-from-eo data
        showing partial coverage

    df_missing :
        data frame indicating grid-cells for
        which both value-eo and value-gb is
        missing

    Returns
    -------
    :
        return predictions for missing
        lat x lon combinations (grid cells)
    """
    target_variables = ["lat", "lon", "year", "season_sin", "season_cos"]

    X_train = df_partial_coverage[target_variables]
    y_train = df_partial_coverage["value_gb"]

    # fit model based on given data
    model_xgb = XGBRegressor(n_estimators=200)
    model_xgb.fit(X_train, y_train)

    # predict based on fitted model
    df_missing["value_gb"] = model_xgb.predict(df_missing[target_variables])
    df_missing["obs_gb"] = False

    return df_missing[["date", "year", "month", "lat", "lon", "value_gb", "obs_gb"]]


def fit_prediction_model(
    df_coverage: pd.DataFrame,
    min_year_pred: int,
    trend_features: list[str],
    resid_features: list[str],
) -> Any:
    """
    Fit model for prediction task

    Parameters
    ----------
    df_coverage :
        full coverage dataset

    min_year_pred :
        minimum year used to inform prediction

    trend_features :
        features/predictors used to estimate trend

    resid_features :
        features/predictors used to estimate
        seasonality, irregularities etc.

    Returns
    -------
    :
        fitted trend_model and residual_model
    """
    df_coverage_processed = preprocessing.preprocess_prediction_dataset(df_coverage)
    df_coverage_processed = df_coverage_processed[
        df_coverage_processed.year > min_year_pred
    ]

    X_linear = df_coverage_processed[trend_features]
    y_actual = df_coverage_processed["value_gb"]

    trend_model = linear_model.LinearRegression()
    trend_model.fit(X_linear, y_actual)

    df_coverage_processed["linear_trend"] = trend_model.predict(X_linear)
    df_coverage_processed["residuals"] = (
        df_coverage_processed["value_gb"] - df_coverage_processed["linear_trend"]
    )

    train_data = lgb.Dataset(
        df_coverage_processed[resid_features], label=df_coverage_processed["residuals"]
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
    }
    residual_model = lgb.train(params, train_data, num_boost_round=500)

    return trend_model, residual_model


def predict_future_years(  # noqa: PLR0913
    trend_model: Any,
    residual_model: Any,
    trend_features: list[str],
    resid_features: list[str],
    pred_year_range: list[int],
    months: int,
    grid_cell_size: int,
    max_lat: int,
    max_lon: int,
) -> pd.DataFrame:
    """
    Predict GHG values for future years

    Parameters
    ----------
    trend_model :
        fitted trend_model

    residual_model :
        fitted residual_model

    trend_features :
        features/predictors used for trend_model

    resid_features :
        features/predictors used for residual_model

    pred_year_range :
        min/max year used for predicting future years

    months :
        number of months

    grid_cell_size :
        size of a single grid-cell

    max_lat :
        maximum absolute latitudinal value

    max_lon :
        maximum absolute longitudinal value

    Returns
    -------
    :
        dataframe for future years with predicted
        GHG values in a variable: "value_gb_pred"
    """
    future_template_data = pd.DataFrame(
        itertools.product(
            np.arange(pred_year_range[0], pred_year_range[1], 1),  # years
            np.arange(1, months, 1),  # months
            np.arange(
                -(max_lat - (grid_cell_size / 2)), max_lat, grid_cell_size
            ),  # latitudes
            np.arange(
                -(max_lon - (grid_cell_size / 2)), max_lon, grid_cell_size
            ),  # longitudes
        ),
        columns=["year", "month", "lat", "lon"],
    )

    future_data_grid = preprocessing.preprocess_prediction_dataset(future_template_data)

    trend_prediction = trend_model.predict(future_data_grid[trend_features])
    resid_prediction = residual_model.predict(future_data_grid[resid_features])

    future_data_grid["value_gb_pred"] = trend_prediction + resid_prediction

    return future_data_grid
