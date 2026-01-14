"""Analysis pipeline for incorporating Earth observations into ground-based data."""

import logging
from typing import Any, Optional

import bambi as bmb  # type: ignore [import-untyped]
import numpy as np
import pandas as pd
from prophet import Prophet  # type: ignore [import-untyped]
from sklearn.metrics import (  # type: ignore [import-untyped]
    mean_absolute_error,
    mean_squared_error,
)

from ghg_forcing_for_cmip import preprocessing

# Mute the CmdStanPy backend (for prediction model)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Mute Prophet's own logger (for prediction model)
logging.getLogger("prophet").setLevel(logging.WARNING)


def fit_gb_from_eo(  # noqa: PLR0913
    formula: str,
    df: pd.DataFrame,
    seed: int,
    categorical: list[str],
    draws: int = 1000,
    chains: int = 4,
    cores: Optional[int] = None,
    target_accept: float = 0.95,
) -> Any:
    """Fit Bayesian regression model to predict ground-based values from EO.

    Parameters
    ----------
    formula
        Statistical model formulation.

    df
        Data used for fitting.

    seed
        Random seed for reproducibility.

    categorical
        Variables in ``formula`` treated as categorical.

    draws
        Posterior draws for MCMC sampling.

    chains
        Number of MCMC chains.

    cores
        CPU cores used for sampling. If ``None``, uses available CPUs.

    target_accept
        Target acceptance rate for NUTS sampler.

    Returns
    -------
    Any
        Tuple of the fitted model and inference data.
    """
    model = bmb.Model(formula=formula, data=df, categorical=categorical)

    idata = model.fit(
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=seed,
        target_accept=target_accept,
        inference_method="numpyro",
    )

    return model, idata


def predict_gb(  # noqa: PLR0913
    idata: Any,
    model: Any,
    in_sample_predictions: bool,
    test_data: Optional[pd.DataFrame] = None,
    n_datasets: Optional[int] = None,
    seed: Optional[int] = None,
    dv_name: str = "value_gb",
) -> Any:
    """Generate posterior predictions of the dependent variable.

    Parameters
    ----------
    idata
        Inference data object from MCMC sampling.

    model
        Fitted Bayesian model.

    in_sample_predictions
        Whether to predict on training data. If ``False``, ``test_data``
        must be provided.

    test_data
        New data for out-of-sample predictions. Required if
        ``in_sample_predictions`` is ``False``.

    n_datasets
        Number of posterior predictive datasets to generate for
        out-of-sample predictions. Required if ``in_sample_predictions``
        is ``False``.

    seed
        Random seed for reproducibility. Required if
        ``in_sample_predictions`` is ``False``.

    dv_name
        Dependent variable name in the model.

    Returns
    -------
    :
        predicted values.
    """
    if not in_sample_predictions and test_data is None:
        msg = "If in_sample_predictions is False, test_data must be provided."
        raise ValueError(msg)

    if in_sample_predictions:
        model.predict(idata, kind="response")

        return idata.posterior_predictive[dv_name].mean(dim=("chain", "draw")).values

    else:
        assert (  # noqa: S101
            n_datasets is not None
        ), "n_datasets must be provided for out-of-sample predictions."
        assert seed is not None, "seed must be provided for out-of-sample predictions."  # noqa: S101
        assert (  # noqa: S101
            test_data is not None
        ), "test_data must be provided for out-of-sample predictions."

        model.predict(idata, data=test_data, kind="response", sample_new_groups=True)

        # get N datasets from posterior draws
        stacked = idata.posterior_predictive["value_gb"].stack(sample=("chain", "draw"))

        total_samples = stacked.sizes["sample"]
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(total_samples, size=n_datasets, replace=False)

        subset = stacked.isel(sample=indices)

        list_of_datasets = []

        for i in range(n_datasets):
            realization = subset.isel(sample=i)
            current_df = test_data.copy()
            current_df["value_gb"] = realization.to_dataframe()["value_gb"].reset_index(
                drop=True
            )
            current_df["dataset"] = i

            list_of_datasets.append(current_df)

        return pd.concat(list_of_datasets)


class PredictionModel:
    """Prophet-based model for predicting future GHG concentrations.

    Fits separate Prophet models for Southern, Tropical, and Northern
    regions based on latitude thresholds. The model configuration
    differs between CO2 and CH4 gases.

    Parameters
    ----------
    gas
        Greenhouse gas type, either "co2" or "ch4".

    Methods
    -------
    __call__(df_coverage, future_time_range, split_value, day=15)
        Main prediction method that orchestrates preprocessing,
        fitting, and postprocessing.
    preprocess(df_coverage, future_time_range, day)
        Prepare data and initialize Prophet models.
    fit_predict(df_prophet, df_future, m_s, m_t, m_n, split_value)
        Fit models and generate forecasts for each region.
    postprocess(df_future, forecast_s, forecast_n, forecast_t, split_value)
        Combine regional forecasts into a single DataFrame.
    """

    def __init__(self, gas: str):
        """Initialize the PredictionModel with a gas type.

        Parameters
        ----------
        gas
            Greenhouse gas type, either "co2" or "ch4".
        """
        self.gas = gas

    def __call__(
        self,
        df_coverage: pd.DataFrame,
        future_time_range: tuple[int, int],
        split_value: int,
        n_datasets: int,
        day: int = 15,
    ) -> pd.DataFrame:
        """Generate predictions for future time periods.

        Parameters
        ----------
        df_coverage
            DataFrame with full spatial coverage of historical GHG data.
        future_time_range
            Tuple of (start_year, end_year) for prediction period.
        split_value
            Latitude threshold for dividing regions. Southern region:
            lat < -split_value, Northern region: lat > split_value,
            Tropical region: -split_value <= lat <= split_value.
        n_datasets
            Number of posterior predictive datasets to generate future
            predictions for.
        day
            Day of month used for creating date variable. Default is 15.

        Returns
        -------
        pd.DataFrame
            DataFrame with predicted GHG values in ``value_gb_pred``
            column for the future time range.
        """
        df_results = []
        for i in range(n_datasets):
            df_prophet, df_future, m_s, m_t, m_n = self.preprocess(
                df_coverage[df_coverage.dataset.isin([i, np.nan])],
                future_time_range,
                day,
            )
            forecast_s, forecast_n, forecast_t = self.fit_predict(
                df_prophet, df_future, m_s, m_t, m_n, split_value
            )
            df_result = self.postprocess(
                df_future, forecast_s, forecast_n, forecast_t, split_value
            )
            df_result["dataset"] = i
            df_results.append(df_result)
        return pd.concat(df_results)

    def preprocess(
        self, df_coverage: pd.DataFrame, future_time_range: tuple[int, int], day: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any, Any]:
        """Prepare data and initialize Prophet models for each region.

        Parameters
        ----------
        df_coverage
            DataFrame with full spatial coverage of historical GHG data.
        future_time_range
            Tuple of (start_year, end_year) for prediction period.
        day
            Day of month used for creating date variable.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, Prophet, Prophet, Prophet]
            Tuple containing:
            - DataFrame prepared for Prophet (with 'ds' and 'y' columns)
            - Future DataFrame for predictions
            - Prophet model for Southern region
            - Prophet model for Tropical region
            - Prophet model for Northern region
        """
        df_future = preprocessing.create_future_dataset(
            future_time_range, lat_only=False
        )
        df_future["ds"] = pd.to_datetime(df_future[["year", "month"]].assign(day=day))

        df_prophet = df_coverage.rename(columns={"date": "ds", "value_gb": "y"})

        # Fit separate models for Southern, Tropical, and Northern regions
        if self.gas == "co2":
            m_s = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
                seasonality_prior_scale=30.0,
            )
            m_t = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
            )
            m_n = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
            )
        elif self.gas == "ch4":
            m_s = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
                seasonality_prior_scale=60.0,
                changepoint_prior_scale=0.5,
                changepoint_range=0.95,
            )
            m_t = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
                changepoint_prior_scale=0.5,
                changepoint_range=0.95,
            )
            m_n = Prophet(
                weekly_seasonality=False,
                daily_seasonality=False,
                yearly_seasonality=True,
                growth="linear",
                changepoint_prior_scale=0.5,
                changepoint_range=0.95,
            )
        else:
            raise ValueError(f"Unsupported gas type: {self.gas}")  # noqa: TRY003

        m_s.add_regressor("lat", standardize=True)
        m_s.add_regressor("lon", standardize=True)

        m_t.add_regressor("lat", standardize=True)
        m_t.add_regressor("lon", standardize=True)

        m_n.add_regressor("lat", standardize=True)
        m_n.add_regressor("lon", standardize=True)

        return df_prophet, df_future, m_s, m_t, m_n

    def fit_predict(  # noqa: PLR0913
        self,
        df_prophet: pd.DataFrame,
        df_future: pd.DataFrame,
        m_s: Any,
        m_t: Any,
        m_n: Any,
        split_value: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit Prophet models and generate forecasts for each region.

        Parameters
        ----------
        df_prophet
            Historical data prepared for Prophet (with 'ds' and 'y' columns).
        df_future
            Future data for predictions (with 'ds' column).
        m_s
            Prophet model for Southern region.
        m_t
            Prophet model for Tropical region.
        m_n
            Prophet model for Northern region.
        split_value
            Latitude threshold for dividing regions.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing forecast DataFrames for Southern, Northern,
            and Tropical regions.
        """
        m_s.fit(df_prophet[df_prophet.lat < -split_value])
        m_n.fit(df_prophet[df_prophet.lat > split_value])
        m_t.fit(
            df_prophet[(df_prophet.lat > -split_value) & (df_prophet.lat < split_value)]
        )

        forecast_s = m_s.predict(df_future[df_future.lat < -split_value])
        forecast_n = m_n.predict(df_future[df_future.lat > split_value])
        forecast_t = m_t.predict(
            df_future[(df_future.lat > -split_value) & (df_future.lat < split_value)]
        )
        return forecast_s, forecast_n, forecast_t

    def postprocess(
        self,
        df_future: pd.DataFrame,
        forecast_s: pd.DataFrame,
        forecast_n: pd.DataFrame,
        forecast_t: pd.DataFrame,
        split_value: int,
    ) -> pd.DataFrame:
        """Combine regional forecasts into a single DataFrame.

        Parameters
        ----------
        df_future
            Future data DataFrame with spatial coordinates.
        forecast_s
            Forecast DataFrame for Southern region.
        forecast_n
            Forecast DataFrame for Northern region.
        forecast_t
            Forecast DataFrame for Tropical region.
        split_value
            Latitude threshold used for region division.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with predicted values in ``value_gb_pred``
            column, assigned based on latitude regions.
        """
        df_future.loc[df_future.lat < -split_value, "value_gb_pred"] = forecast_s[
            "yhat"
        ].values
        df_future.loc[df_future.lat > split_value, "value_gb_pred"] = forecast_n[
            "yhat"
        ].values
        df_future.loc[
            (df_future.lat > -split_value) & (df_future.lat < split_value),
            "value_gb_pred",
        ] = forecast_t["yhat"].values
        return df_future


def evaluate_prediction_model(
    model: Any,
    df_full: pd.DataFrame,
    cutoffs: list[int],
    split_value: int,
    future_years: int = 1,
) -> pd.DataFrame:
    """
    Perform temporal cross-validation to assess forecast performance.

    Parameters
    ----------
    model : PredictionModel
        Instance of the prediction model class.
    df_full : pd.DataFrame
        Complete historical dataset containing 'year', 'value_gb', etc.
    cutoffs : list[int]
        List of years to use as split points for training/testing.
    split_value : int
        Latitude threshold for regional division in the model.
    future_years : int
        Duration of the forecast horizon to evaluate (default: 1 year).

    Returns
    -------
    pd.DataFrame
        Table of RMSE and MAE metrics for each cutoff year.
    """
    metrics = []

    for cutoff_year in cutoffs:
        df_train = df_full[df_full["year"] < cutoff_year].copy()

        df_test = df_full[
            (df_full["year"] >= cutoff_year)
            & (df_full["year"] < cutoff_year + future_years)
        ].copy()

        pred_df = model(
            df_coverage=df_train,
            future_time_range=(cutoff_year, cutoff_year + future_years),
            split_value=split_value,
            n_datasets=1,
        )

        comparison = df_test[["year", "month", "lat", "lon", "value_gb"]].merge(
            pred_df[["year", "month", "lat", "lon", "value_gb_pred"]],
            on=["year", "month", "lat", "lon"],
        )

        y_true = comparison["value_gb"]
        y_pred = comparison["value_gb_pred"]

        metrics.append(
            {
                "cutoff": cutoff_year,
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "n_samples": len(comparison),
            }
        )

    return pd.DataFrame(metrics)


def extract_prophet_components(
    model: PredictionModel,
    df_coverage: pd.DataFrame,
    future_time_range: tuple[int, int],
    split_value: int,
    day: int = 15,
) -> dict[str, pd.DataFrame]:
    """Extract trend and seasonality components from Prophet models.

    Fits Prophet models and extracts the trend and yearly seasonality
    components for visualization purposes.

    Parameters
    ----------
    model
        Instance of PredictionModel.
    df_coverage
        Historical data for fitting the models.
    future_time_range
        Tuple of (start_year, end_year) for the prediction period.
    split_value
        Latitude threshold for dividing regions.
    day
        Day of month used for creating date variable.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys 'southern', 'tropical', 'northern', each
        containing a DataFrame with columns 'ds', 'trend', 'yearly'.
    """
    # Use first dataset if multiple exist
    df_subset = df_coverage[df_coverage.dataset.isin([0, np.nan])].copy()

    df_prophet, df_future, m_s, m_t, m_n = model.preprocess(
        df_subset, future_time_range, day
    )

    # Fit models
    m_s.fit(df_prophet[df_prophet.lat < -split_value])
    m_n.fit(df_prophet[df_prophet.lat > split_value])
    m_t.fit(
        df_prophet[(df_prophet.lat > -split_value) & (df_prophet.lat < split_value)]
    )

    # Create extended dataframe for component extraction
    # Combine historical and future periods
    df_extended = (
        pd.concat(
            [df_prophet[["ds"]].drop_duplicates(), df_future[["ds"]].drop_duplicates()]
        )
        .drop_duplicates()
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Get components for each region
    # For each region, we need representative lat/lon values
    components = {}

    for region_name, prophet_model, region_filter in [
        ("southern", m_s, lambda x: x.lat < -split_value),
        ("tropical", m_t, lambda x: (x.lat > -split_value) & (x.lat < split_value)),
        ("northern", m_n, lambda x: x.lat > split_value),
    ]:
        # Get representative coordinates for this region
        region_data = df_prophet[region_filter(df_prophet)]
        if len(region_data) == 0:
            continue

        # Use median lat/lon for the region
        rep_lat = region_data["lat"].median()
        rep_lon = region_data["lon"].median()

        # Create dataframe for prediction
        df_pred = df_extended.copy()
        df_pred["lat"] = rep_lat
        df_pred["lon"] = rep_lon

        # Predict to get components
        forecast = prophet_model.predict(df_pred)

        # Extract components
        # Prophet includes 'trend' and 'yearly'
        # (since yearly_seasonality=True in all models)
        components[region_name] = pd.DataFrame(
            {
                "ds": forecast["ds"],
                "trend": forecast["trend"],
                "yearly": forecast["yearly"],
            }
        )

    return components
