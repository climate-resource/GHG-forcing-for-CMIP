"""
plotting functions for documentation in mkdocs
"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp  # type: ignore


def plot_predictions(
    predictions: tf.Tensor, X_time: tf.Tensor, y_obs: pd.Series, ghg: str, prior: bool
) -> None:
    """
    Plot predictions (either prior or posterior predictions)

    Parameters
    ----------
    predictions :
        tensor with prior or posterior predictions

    X_time :
        time variable

    y_obs :
        observed data

    ghg :
        target greenhouse gas variable

    prior :
        whether prior or posterior predictions are plotted
        configures the time axis and plot labels
    """
    if prior:
        label = "prior predictions"
        X_time_obs = X_time
    else:
        label = "posterior predictions"
        X_time_obs = X_time[: len(y_obs)]

    pred_sd = tf.math.reduce_std(predictions, 0)
    pred_median = tfp.stats.percentile(predictions, 50, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.fill_between(
        X_time[:, 0],
        pred_median - pred_sd,
        pred_median + pred_sd,
        alpha=0.8,
        color="lightblue",
    )
    ax.plot(X_time, pred_median, lw=2, color="darkblue", label=label)
    if not prior:
        ax.fill_between(
            X_time[len(y_obs) :, 0],
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
    ax.plot(X_time_obs, y_obs, color="red", lw=2, label="observed")
    ax.legend(handlelength=0.3, frameon=False)
    ax.set_xlabel("time (year-month)")
    ax.set_ylabel(ghg)
    ax.tick_params(axis="both", labelsize=8)
    plt.show()
