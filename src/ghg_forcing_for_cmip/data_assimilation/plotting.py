"""
plotting functions for documentation in mkdocs
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore


def plot_predictions(  # noqa: PLR0913
    predictions: tf.Tensor,
    X_time_obs: tf.Tensor,
    X_time_pred: tf.Tensor,
    y_obs: pd.Series,
    d_group_obs: pd.Series,
    d_group_pred: pd.Series,
    ghg: str,
    prior: str,
    save_fig: bool = True,
) -> None:
    """
    Plot predictions (either prior or posterior predictions)

    Parameters
    ----------
    predictions :
        tensor with prior or posterior predictions

    X_time_obs :
        observed time variable

    X_time_pred :
        predicted time variable

    y_obs :
        observed data

    d_group :
        observed group data

    X_group_pred :
        predicted group variable

    ghg :
        target greenhouse gas variable

    prior :
        whether prior or posterior predictions are plotted
        configures the time axis and plot labels

    save_fig:
        whether to save the figure
    """
    pred_sd = tf.math.reduce_std(predictions, 0)
    pred_median = tfp.stats.percentile(predictions, 50, axis=0)

    df_obs = pd.DataFrame(
        dict(time=X_time_obs, y_obs=y_obs, group=d_group_obs)
    ).replace({f"x{ghg}": "EO (obs)", "value_vertical": "GB (obs)"})

    df_pred = pd.DataFrame(
        dict(time=X_time_pred, y_mod=pred_median, group=d_group_pred)
    ).replace(
        {f"x{ghg}": f"EO ({prior} pred.)", "value_vertical": f"GB ({prior} pred.)"}
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    sns.lineplot(
        data=df_obs,
        x="time",
        y="y_obs",
        hue="group",
        linestyle=(0, (1, 1)),
        lw=2,
        ax=ax,
        zorder=1,
        errorbar=("pi", 100),
    )
    ax.fill_between(
        df_pred["time"],
        pred_median - pred_sd,
        pred_median + pred_sd,
        alpha=0.8,
        color="lightblue",
        zorder=0,
    )
    sns.lineplot(
        data=df_pred,
        x="time",
        y="y_mod",
        hue="group",
        lw=2,
        ax=ax,
        zorder=0,
        errorbar=("pi", 100),
    )

    ax.legend(handlelength=0.8, frameon=False, ncol=2)
    ax.set_xlabel("time (year-month)")
    ax.set_ylabel(rf"$X{ghg.upper().replace(ghg[-1], f'_{ghg[-1]}')}$")
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if save_fig:
        plt.savefig(f"report/plots/{ghg}_{prior}_predictions.png")
    plt.show()


def plot_predictions_source(
    predictions: tf.Tensor, X_time: tf.Tensor, y_obs: pd.Series, ghg: str, prior: bool
) -> None:
    """
    Plot prior or posterior predictions

    Parameters
    ----------
    predictions :
        prior/posterior predictions are returned by the prob. model

    X_time:
        design matrix for variable time

    y_obs:
        observed GHG concentration values

    ghg :
        target greenhouse gas variable

    prior :
        if true, prior predictions are plotted
        if false, posterior predictions are plotted
    """
    if prior:
        label = "prior predictions"
        X_time_obs = X_time
    else:
        label = "posterior predictions"
        X_time_obs = X_time[: len(y_obs)]

    source = ["satellite", "ground-based"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    for i, (sou, linesty) in enumerate(zip(source, [":", "-"])):
        pred_sd = tf.math.reduce_std(predictions[:, i::2], 0)
        pred_median = tfp.stats.percentile(predictions[:, i::2], 50, axis=0)

        ax.fill_between(
            X_time,
            pred_median - pred_sd,
            pred_median + pred_sd,
            alpha=0.8,
            color="lightblue",
        )
        ax.plot(
            X_time, pred_median, lw=1, color="darkblue", label=label, linestyle=linesty
        )
        if not prior:
            ax.fill_between(
                X_time[len(y_obs) :],
                pred_median[len(y_obs) :] - pred_sd[len(y_obs) :],
                pred_median[len(y_obs) :] + pred_sd[len(y_obs) :],
                alpha=0.8,
                color="lightgray",
            )
            ax.plot(
                X_time[len(y_obs) :],
                pred_median[len(y_obs) :],
                lw=2,
                color="darkgrey",
                label="forecast",
            )
        ax.plot(
            X_time_obs, y_obs[i::2], color="red", lw=1, label=sou, linestyle=linesty
        )
    ax.legend(handlelength=0.6, frameon=False)
    ax.set_xlabel("time (year-month)")
    ax.set_ylabel(ghg)
    ax.tick_params(axis="both", labelsize=8)
    plt.show()


def plot_posterior_source(
    predictions: tf.Tensor,
    X_time_pred: tf.Tensor,
    y_obs: pd.Series,
    ghg: str,
) -> None:
    """
    Plot posterior predictions for dataset incl. source variable

    Parameters
    ----------
    predictions :
        posterior predictions are returned by the prob. model

    X_time:
        design matrix for variable time

    y_obs:
        observed GHG concentration values

    ghg :
        target greenhouse gas variable
    """
    source = ["satellite", "ground-based"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    # Store handles and labels manually
    custom_handles = {}
    linestyles = [":", "-"]
    alphas = [0.2, 0.6]

    for i, (sou, linesty, alp) in enumerate(zip(source, linestyles, alphas)):
        pred_sd = tf.math.reduce_std(predictions[:, i::2], 0)
        pred_median = tfp.stats.percentile(predictions[:, i::2], 50, axis=0)

        # Main prediction interval
        ax.fill_between(
            X_time_pred[::2],
            pred_median - pred_sd,
            pred_median + pred_sd,
            alpha=0.8,
            color="lightblue",
        )
        # Prediction line
        (pred_line,) = ax.plot(
            X_time_pred[::2],
            pred_median,
            lw=1,
            color="darkblue",
            linestyle=linesty,
            label=None,  # don't include in legend
        )

        (forecast_line,) = ax.plot(
            X_time_pred[::2][len(y_obs[i::2]) :],
            pred_median[len(y_obs[i::2]) :],
            lw=1,
            color="darkgrey",
            linestyle=linesty,
            label=None,
        )
        # Add only one forecast label to the legend
        if "forecast" not in custom_handles:
            custom_handles["forecast"] = forecast_line

        # Observed data
        (obs_line,) = ax.plot(
            X_time_pred[: len(y_obs)][i::2],
            y_obs[i::2],
            color="red",
            lw=1,
            label=None,
            linestyle=linesty,
        )
        if sou not in custom_handles:
            custom_handles[sou] = obs_line

    # Custom legend
    ax.legend(
        custom_handles.values(), custom_handles.keys(), handlelength=0.6, frameon=False
    )

    ax.set_xlabel("time (year-month)")
    ax.set_ylabel(ghg)
    ax.tick_params(axis="both", labelsize=8)
    plt.show()


def plot_seasonality(df: pd.DataFrame, value: str) -> None:
    """
    Plot seasonality from predicted and observed values

    Parameters
    ----------
    df :
        dataframe combining predicted and observed values

    value :
        name of dependent variable; either "value" (for
        observed values) or "ypred_mean" (for predicted
        values)
    """
    df_season = df.groupby(["month", "lat"]).agg({value: "mean"}).reset_index()

    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        df_season.pivot(index="lat", columns="month", values=value),
        yticklabels=True,
        cmap="crest",
    )
    ax.invert_yaxis()
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("$CH_4$ in ppb")
    plt.yticks(size=7)
    plt.xticks(size=7, rotation=45)
    plt.xlabel("month")
    plt.ylabel("latitude")
    if value == "value":
        plt.title("Seasonality (observed)")
    else:
        plt.title("Seasonality (predicted)")
    plt.show()


def plot_annual(df: pd.DataFrame, value: str) -> None:
    """
    Plot monthly average from predicted and observed values

    Parameters
    ----------
    df :
        dataframe combining predicted and observed values

    value :
        name of dependent variable; either "value" (for
        observed values) or "ypred_mean" (for predicted
        values)
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 4))

    heat = sns.heatmap(
        df.pivot(index="lat", columns="year_month", values=value), cmap="crest"
    )
    heat.invert_yaxis()
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("$CH_4$ in ppb")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.set_ylabel("latitude")
    ax.set_xlabel("time (year-month)")
    if value == "value":
        ax.set_title("monthly average ($y_{obs}$)")
    else:
        ax.set_title("monthly average ($y_{pred}$)")
    plt.show()


def plot_error(df: pd.DataFrame) -> None:
    """
    Plot error between observed and predicted values

    Error computed as root squared error

    Parameters
    ----------
    df:
        dataframe combining predicted and observed values
        (should not include predicted years)
    """
    df["error"] = np.sqrt(abs(df.value - df.ypred_mean) ** 2)

    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        df.pivot(index="lat", columns="year_month", values="error"),
        yticklabels=True,
        cmap="crest",
    )
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("$|y_{pred} - y_{obs}|$")
    ax.invert_yaxis()
    ax.set_ylabel("latitude")
    ax.set_xlabel("time (year/month)")
    plt.title("Error: observed vs. predicted $CH_4$ in ppb")
    plt.yticks(size=7)
    plt.xticks(size=7, rotation=45)
    plt.show()
