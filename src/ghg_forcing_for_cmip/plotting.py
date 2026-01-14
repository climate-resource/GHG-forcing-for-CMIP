"""
Plotting functions for tutorials in documentation

"""

from typing import Any, Optional

import geodatasets  # type: ignore[import-untyped]
import geopandas  # type: ignore[import-untyped]
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

from ghg_forcing_for_cmip import preprocessing
from ghg_forcing_for_cmip.exceptions import MissingOptionalDependencyError
from ghg_forcing_for_cmip.validation import compute_discrepancy_collocated


def plot_map(  # noqa: PLR0913
    d: pd.DataFrame,
    title: str,
    axs: Any,
    lon_value: str = "longitude",
    lat_value: str = "latitude",
    marker: str = "x",
    markersize: int = 20,
    figsize: tuple[int, int] = (8, 3),
) -> Any:
    """
    Plot world map with observation stations

    Parameters
    ----------
    d :
        dataframe with ghg concentrations and lat, lon information

    title :
        title of the plot

    axs :
        axes of matplotlib

    lon_value :
        name of longitudinal variable (column)

    lat_value :
        name of latitudinal variable (column)

    marker :
        shape of plot markers

    markersize :
        size of plot markers

    figsize :
        size of world map

    Returns
    -------
    :
        axes from matplotlib pyplot
    """
    try:
        import geopandas
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="geopandas"
        ) from exc

    try:
        from geodatasets import get_path
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="geodatasets"
        ) from exc

    gdf = geopandas.GeoDataFrame(
        d,
        geometry=geopandas.points_from_xy(d[lon_value], d[lat_value]),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(get_path("naturalearth.land"), engine="fiona")
    world.plot(ax=axs, color="white", edgecolor="grey")
    gdf.plot(figsize=figsize, ax=axs, color="red", marker=marker, markersize=markersize)
    axs.set_title(title)
    return axs


def plot_monthly_average(
    d: pd.DataFrame,
    gas: str,
    axs: Any,
    label_eo: str = "satellite",
    label_gb: str = "ground_based",
) -> Any:
    """
    Plot global, monthly average of GHG concentration

    Parameters
    ----------
    d :
        dataframe with raw data

    gas :
        either "ch4" or "co2"

    axs :
        axis of plt.subplots object

    label_eo :
        legend in plot for earth observations,
        by default "satellite"

    label_gb :
        legend in plot for ground-based data,
        by default "ground_based"

    Returns
    -------
    :
        axis object from matplotlib.pyplot
    """
    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    d["time_fractional"] = d.year + d.month / 12
    d_avg = (
        d.groupby(["year", "month"])
        .agg({"value_gb": "mean", "value_eo": "mean", "time_fractional": "mean"})
        .reset_index()
    )

    sns.lineplot(data=d_avg, x="time_fractional", y="value_eo", ax=axs, label=label_eo)
    sns.lineplot(data=d_avg, x="time_fractional", y="value_gb", ax=axs, label=label_gb)
    axs.set_xlabel("time")
    axs.set_ylabel(gas)
    axs.legend(frameon=False)
    axs.spines["right"].set_visible(False)
    axs.spines["top"].set_visible(False)
    return axs


def plot_average_hemisphere(d_colloc: pd.DataFrame, gas: str) -> None:
    """
    Plot global, monthly average separate for hemispheres

    Parameters
    ----------
    d_colloc :
        dataframe including grouping variable with
        hemisphere levels (Northern, Southern, Tropics)
        whereby Northern: >30°, Southern: < -30°, and
        Tropics: <30°, >-30°

    gas :
        either "co2" or "ch4"
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    conditions = [d_colloc["lat"] > 30, d_colloc["lat"] < -30]  # noqa: PLR2004
    d_colloc["hemisphere"] = np.select(
        conditions, ["Northern", "Southern"], default="Tropics"
    )

    _, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(11, 3), sharey=True)
    d_avg = (
        d_colloc.groupby(["year", "month", "hemisphere"])
        .agg({"value_gb": "mean", "value_eo": "mean", "time_fractional": "mean"})
        .reset_index()
    )
    for i, hemi in enumerate(["Northern", "Southern", "Tropics"]):
        d = d_avg[d_avg.hemisphere == hemi]
        sns.lineplot(
            d,
            x="time_fractional",
            y="value_gb",
            linewidth=2.0,
            ax=axs[i],
            label="ground-based",
        )
        sns.lineplot(
            d,
            x="time_fractional",
            y="value_eo",
            linewidth=2.0,
            ax=axs[i],
            label="satellite",
        )
        axs[i].set_xlabel("time")
        axs[i].set_title(np.where(hemi == "Tropics", hemi, f"{hemi} hemisphere"))
        axs[0].set_ylabel(gas)
        axs[i].legend(frameon=False)


def plot_collocated_rmse(
    d_colloc_co2: pd.DataFrame, d_colloc_ch4: pd.DataFrame, measure: str
) -> Any:
    """
    Plot rmse for collocated data for GB vs. EO data

    Parameters
    ----------
    d_colloc_co2 :
        collocated data for co2

    d_colloc_ch4 :
        collocated data for ch4

    measure :
        either "rmse" or "dcor"

    Returns
    -------
    :
        axs of matplotlib.pyplot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="matplotlib"
        ) from exc

    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError("plotting", requirement="seaborn") from exc

    df_measure = pd.DataFrame(
        compute_discrepancy_collocated(d_colloc_co2, "co2", measure)
    ).merge(
        compute_discrepancy_collocated(d_colloc_ch4, "ch4", measure), on="site_code"
    )
    df_measure["sd_co2"] = np.sqrt(df_measure["var_co2"])
    df_measure["sd_ch4"] = np.sqrt(df_measure["var_ch4"])

    _, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 5))
    for meas in ["rmse", "bias", "sd"]:
        sns.lineplot(
            data=df_measure.sort_values(by=f"{meas}_co2"),
            y=f"{meas}_co2",
            x="site_code",
            ax=axs[0],
            label=meas.upper(),
        )
        sns.lineplot(
            data=df_measure.sort_values(by=f"{meas}_ch4"),
            y=f"{meas}_ch4",
            x="site_code",
            ax=axs[1],
        )
    for i in range(2):
        axs[i].tick_params(axis="both", labelsize=8)
        axs[i].set_xlabel(measure.upper())
        axs[i].tick_params(axis="x", rotation=90)
        axs[i].axhline(0, linestyle="dashed", color="darkgrey", lw=1)
    axs[1].set_ylabel("")
    axs[1].set_title(
        ("Methane (avg. RMSE:" + f"{df_measure[f'{measure}_ch4'].mean():.2f})"),
        fontsize="medium",
    )
    axs[0].set_title(
        ("Carbon Dioxide (avg. RMSE:" + f"{df_measure[f'{measure}_co2'].mean():.2f})"),
        fontsize="medium",
    )
    axs[0].set_ylabel("CO2 in ppm")
    axs[1].set_ylabel("CH4 in ppb")
    return axs


def plot_global_hemisphere(
    df: pd.DataFrame, gas: str, unit: str, figsize: Optional[tuple[int, int]] = None
) -> Any:
    """
    Plot global average of GHG

    grouped by northern, southern hemispheres and tropics

    Parameters
    ----------
    df :
        dataframe

    gas :
        name of greenhouse gas

    unit :
        unit of greenhouse gas

    figsize :
        size of figure

    Returns
    -------
    :
        fig, axs
    """
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(
        data=df, x="date", y="value_gb_pred", hue="hemisphere", lw=2, alpha=0.5, ax=axs
    )
    for hemi, label in zip(df.hemisphere.unique(), ["observed", None, None]):
        sns.scatterplot(
            data=df[df.hemisphere == hemi],
            x="date",
            y="value_gb",
            color="black",
            s=5,
            lw=0,
            ax=axs,
            label=label,
        )
    axs.set_ylabel(f"{gas.upper()} in {unit}")
    axs.spines[["right", "top"]].set_visible(False)
    axs.legend(frameon=False, handlelength=0.5, ncol=3)

    return fig, axs


def plot_global_hemisphere_coverage(  # noqa: PLR0913
    df_collocated: pd.DataFrame,
    df_coverage: pd.DataFrame,
    gas: str,
    unit: str,
    split_value: int,
    figsize: Optional[tuple[int, int]] = None,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> Any:
    """
    Plot global, monthly-avg. ghg concentration

    Plotting for different hemispheres and compared to
    observed data.

    Parameters
    ----------
    df_collocated :
        collocated dataframe including observed data
        (i.e., gb_value)

    df_coverage :
        gap filled dataframe including predicted data
        that covers full spatio-temporal range

    gas :
        name of greenhouse gas

    unit :
        unit of greenhouse gas

    split_value :
        splitting value used to differentiate between
        northern, southern hemisphere and tropics

    figsize :
        size of figure

    quantiles :
        quantiles used for uncertainty shading

    Returns
    -------
    :
        fig, axs
    """
    df_pred = preprocessing.add_hemisphere(df_coverage, split_value=split_value)
    df_pred = (
        df_pred.groupby(["date", "hemisphere"])
        .agg(
            mean=("value_gb", "mean"),
            lower=("value_gb", lambda x: x.quantile(quantiles[0])),
            upper=("value_gb", lambda x: x.quantile(quantiles[1])),
        )
        .reset_index()
    )

    df_observed = preprocessing.add_hemisphere(df_collocated, split_value=split_value)
    df_observed = (
        df_observed.groupby(["date", "hemisphere"])
        .agg({"value_gb": "mean"})
        .reset_index()
    )

    unique_hemis = df_pred["hemisphere"].unique()
    palette = dict(
        zip(unique_hemis, sns.color_palette("tab10", n_colors=len(unique_hemis)))
    )

    fig, axs = plt.subplots(3, 1, figsize=figsize, constrained_layout=True, sharex=True)

    for i, hemi in enumerate(unique_hemis):
        subset = df_pred[df_pred["hemisphere"] == hemi]
        color = palette[hemi]

        axs[i].plot(subset["date"], subset["mean"], color=color, lw=2)
        axs[i].set_title(hemi)

        axs[i].fill_between(
            subset["date"],
            subset["lower"],
            subset["upper"],
            color=color,
            alpha=0.3,
            edgecolor="none",
        )

    for i, (hemi, label) in enumerate(
        zip(df_pred.hemisphere.unique(), ["observed", None, None])
    ):
        sns.scatterplot(
            data=df_observed[df_observed.hemisphere == hemi],
            x="date",
            y="value_gb",
            color="black",
            s=5,
            lw=0,
            ax=axs[i],
            label=label,
            zorder=3,
        )

        axs[i].set_ylabel(f"{gas.upper()} in {unit}")
        axs[i].spines[["right", "top"]].set_visible(False)

    axs[0].legend(frameon=False, handlelength=0.5, ncol=3)

    return fig, axs


def plot_coverage(  # noqa: PLR0913
    df: pd.DataFrame,
    year: int,
    grid_size: int,
    gas: str,
    unit: str,
    ms: int = 10,
    lw: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlim: Optional[list[float]] = None,
    ylim: Optional[list[float]] = None,
) -> Any:
    """
    Plot GHG observations on world map

    Parameters
    ----------
    df :
        dataset including ghg data

    year :
        year for which data should be visualized

    grid_size :
        size of single grid cells

    gas :
        name of greenhouse gas

    unit :
        unit of greenhouse gas

    ms :
        marker size, by default 10

    lw :
        red border used to visualize observed data,
        by default 0.5

    vmin :
        minimum scale value for legend,
        if `None` will be determined by data

    vmax :
        maximum scale value for legend,
        if `None` will be determined by data

    xlim :
        range of x-axis that should be visualized,
        can be used to "zoom" into the map

    ylim :
        range of y-axis that should be visualized,
        can be used to "zoom" into the map

    Returns
    -------
    :
        fig, axs
    """
    # compute grid-cell average
    df_gb_binned = (
        df[df.year == year]
        .groupby(["lat", "lon"])
        .agg({"value_gb": "mean", "obs_gb": "max"})
        .reset_index()
    )

    gdf_gb = geopandas.GeoDataFrame(
        df_gb_binned,
        geometry=geopandas.points_from_xy(df_gb_binned.lon, df_gb_binned.lat),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(geodatasets.get_path("naturalearth.land"))

    legend_kwds = {
        "shrink": 0.8,
        "label": f"{gas.upper()} [{unit}]",
        "orientation": "vertical",
    }

    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    world.plot(ax=axs, facecolor="none", edgecolor="black")

    gdf_gb.plot(
        ax=axs,
        column=df_gb_binned.value_gb,
        marker="s",
        markersize=ms,
        vmin=vmin,
        vmax=vmax,
        zorder=0,
        legend=True,
        legend_kwds=legend_kwds,
    )
    gdf_highlight = gdf_gb[gdf_gb["obs_gb"] == True]  # noqa

    if not gdf_highlight.empty:
        gdf_highlight.plot(
            ax=axs,
            facecolor="none",
            edgecolor="red",
            linewidth=lw,
            marker="s",
            markersize=ms,
            zorder=0,
        )

    for hl in np.arange(0, 90 + grid_size, grid_size):
        axs.axhline(float(hl), color="lightgrey", lw=0.5)
        axs.axhline(-float(hl), color="lightgrey", lw=0.5)
    for vl in np.arange(0, 180 + grid_size, grid_size):
        axs.axvline(float(vl), color="lightgrey", lw=0.5)
        axs.axvline(-float(vl), color="lightgrey", lw=0.5)

    if xlim is not None:
        axs.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])

    axs.set_title(f"{grid_size} x {grid_size} grid size (red squares: observed GB)")
    axs.set_xlabel("longitude")
    axs.set_ylabel("latitude")

    return fig, axs


def plot_hemisphere(  # noqa: PLR0913
    df_pred: pd.DataFrame,
    df_obs: pd.DataFrame,
    gas: str,
    unit: str,
    year_min: int,
    split_value: int,
    figsize: tuple[int],
    day: int = 15,
) -> Any:
    """
    Plot observed and predicted data aggregated over different hemispheres

    Parameters
    ----------
    df_pred :
        predicted data, spanning full coverage

    df_obs :
        observed ground-based data

    gas :
        name of greenhouse gas

    unit :
        unit of greenhouse gas measurement

    year_min :
        minimum year used for showing x-range

    split_value :
        value used to differentiate between
        northern, southern hemisphere and tropics

    figsize :
        plot size

    day :
        day used for constructing date variable

    Returns
    -------
    Any
        fig, axs
    """
    df_coverage = preprocessing.add_hemisphere(df_pred, split_value=split_value)
    df_observed = preprocessing.add_hemisphere(df_obs, split_value=split_value)
    df_observed["date"] = pd.to_datetime(df_observed[["year", "month"]].assign(day=day))
    df_observed = (
        df_observed[df_observed.year > year_min]
        .groupby(["date", "hemisphere"])
        .agg({"value": "mean"})
        .reset_index()
    )

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(
        data=df_coverage.groupby(["date", "hemisphere"])
        .agg({"value_gb": "mean"})
        .reset_index(),
        x="date",
        y="value_gb",
        hue="hemisphere",
        ax=axs,
        lw=3,
        alpha=0.4,
    )
    for hemi in df_observed.hemisphere.unique():
        sns.scatterplot(
            data=df_observed[df_observed.hemisphere == hemi],
            x="date",
            y="value",
            color="black",
            ax=axs,
            lw=0,
            s=5,
        )
    axs.legend(frameon=False, handlelength=0.5, ncol=3)
    axs.set_ylabel(f"{gas.upper()} in {unit}")
    axs.spines[["right", "top"]].set_visible(False)

    return fig, axs


def plot_future_predictions(  # noqa: PLR0913
    df_observed: pd.DataFrame,
    df_coverage: pd.DataFrame,
    df_future_years: pd.DataFrame,
    min_year: int,
    split_value: int,
    figsize: tuple[int],
    gas: str,
    unit: str,
    day: int = 15,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> Any:
    """
    Plot observed, fitted, and predicted data

    Monthly aggregated across hemispheres and for the global mean.

    Parameters
    ----------
    df_observed :
        observed GHG data

    df_coverage :
        predicted data within observed year range showing full coverage

    df_future_years :
        predicted data for future years

    min_year :
        minimum year for truncating the x-axis

    split_value :
        value used to differentiate between southern,
        northern hemisphere, and tropics

    figsize :
        size of plot

    gas :
        name of GHG

    unit :
        unit of GHG

    day :
        day used for creating a date variable

    quantiles :
        quantiles used for uncertainty shading

    Returns
    -------
    :
        fig, axs
    """

    def add_date(df: pd.DataFrame) -> pd.DataFrame:
        """Assign date variable"""
        return df.assign(date=pd.to_datetime(df[["year", "month"]].assign(day=day)))

    df_full = df_coverage[df_coverage.year > min_year].copy()
    df_full = preprocessing.add_hemisphere(df_full, split_value)
    df_full = add_date(df_full)

    df_future_years = df_future_years[df_future_years.year > min_year].copy()
    df_future_years = preprocessing.add_hemisphere(df_future_years, split_value)
    df_future_years = add_date(df_future_years)

    df_observed = df_observed[df_observed.year > min_year].copy()
    df_observed = preprocessing.add_hemisphere(df_observed, split_value)
    df_observed = add_date(df_observed)

    dfs_hemi = []
    dv_names = ["value_gb", "value_gb_pred", "value"]

    input_dfs = [df_full, df_future_years, df_observed]

    for df, dv_name in zip(input_dfs, dv_names):
        dfs_hemi.append(
            df.groupby(["date", "hemisphere"])
            .agg(
                mean=(dv_name, "mean"),
                lower=(dv_name, lambda x: x.quantile(quantiles[0])),
                upper=(dv_name, lambda x: x.quantile(quantiles[1])),
            )
            .reset_index()
        )

    fig, axs = plt.subplots(3, 1, figsize=figsize, constrained_layout=True, sharex=True)

    plot_configs = [
        (0, "northern", "tab:blue"),
        (1, "southern", "tab:orange"),
        (2, "tropics", "tab:green"),
    ]

    fit_hemi, fut_hemi, obs_hemi = dfs_hemi

    for c, region, color in plot_configs:
        ax = axs[c]

        df_fit = fit_hemi[fit_hemi.hemisphere == region]
        df_fut = fut_hemi[fut_hemi.hemisphere == region]
        df_obs = obs_hemi[obs_hemi.hemisphere == region]
        title = region.capitalize()

        sns.scatterplot(
            data=df_obs,
            x="date",
            y="mean",
            color="black",
            s=10,
            alpha=0.6,
            ax=ax,
            label="Observed" if c == 0 else None,
        )

        ax.plot(
            df_fit.date,
            df_fit["mean"],
            color=color,
            lw=1,
            label="Fitted" if c == 0 else None,
        )

        ax.plot(
            df_fut.date,
            df_fut["mean"],
            color=color,
            lw=2,
            label="Prediction" if c == 0 else None,
        )
        ax.fill_between(
            df_fut.date,
            df_fut.lower,
            df_fut.upper,
            color=color,
            alpha=0.1,
            edgecolor="none",
        )

        ax.set_title(title)
        ax.set_ylabel(f"{gas.upper()} [{unit}]")
        ax.set_xlabel("")
        ax.spines[["right", "top"]].set_visible(False)

    axs[0].legend(frameon=False, loc="upper left", fontsize="small")
    for i in range(3):
        axs[i].set_xlabel("date")
    return fig, axs


def plot_locations(  # noqa: PLR0913
    df_observed: pd.DataFrame,
    df_fitted: pd.DataFrame,
    year_min: int,
    figsize: tuple[int],
    gas: str,
    unit: str,
    locations: pd.DataFrame,
    df_predicted: Optional[pd.DataFrame] = None,
    day: int = 15,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> Any:
    """
    Plot future years

    Parameters
    ----------
    df_observed :
        observed data set

    df_fitted :
        fitted data to observed data

    year_min :
        minimum year used for truncating the x-axis

    figsize :
        size of figure

    gas :
        name of greenhouse gas

    unit :
        unit of greenhouse gas

    locations :
        selected locations to plot

    df_predicted :
        predict future data

    day :
        day used for creating a date variable, default is 15

    quantiles :
        quantiles used for uncertainty shading

    Returns
    -------
    :
        fig, axs
    """
    df_observed["date"] = pd.to_datetime(df_observed[["year", "month"]].assign(day=day))
    df_fitted["date"] = pd.to_datetime(df_fitted[["year", "month"]].assign(day=day))

    fig, axs = plt.subplots(
        4, 3, constrained_layout=True, sharey=True, sharex=True, figsize=figsize
    )
    k = 0
    for j in range(4):
        for i in range(3):
            observed_loc = df_observed[
                (df_observed.lat == locations.lat[k])
                & (df_observed.lon == locations.lon[k])
                & (df_observed.year > year_min)
            ]
            fitted_loc = df_fitted[
                (df_fitted.lat == locations.lat[k])
                & (df_fitted.lon == locations.lon[k])
                & (df_fitted.year > year_min)
            ]
            fitted_loc = (
                fitted_loc.groupby(["date"])
                .agg(
                    mean=("value_gb", "mean"),
                    lower=("value_gb", lambda x: x.quantile(quantiles[0])),
                    upper=("value_gb", lambda x: x.quantile(quantiles[1])),
                )
                .reset_index()
            )

            if df_predicted is not None:
                df_predicted["date"] = pd.to_datetime(
                    df_predicted[["year", "month"]].assign(day=day)
                )

                predicted_loc = df_predicted[
                    (df_predicted.lat == locations.lat[k])
                    & (df_predicted.lon == locations.lon[k])
                    & (df_predicted.year > year_min)
                ]
                predicted_loc = (
                    predicted_loc.groupby(["date"])
                    .agg(
                        mean=("value_gb_pred", "mean"),
                        lower=("value_gb_pred", lambda x: x.quantile(quantiles[0])),
                        upper=("value_gb_pred", lambda x: x.quantile(quantiles[1])),
                    )
                    .reset_index()
                )

            if (i == 0) & (j == 0):
                label_obs, label_fit, label_pred = "observed", "fitted", "predicted"
            else:
                label_obs, label_fit, label_pred = None, None, None

            sns.scatterplot(
                data=observed_loc,
                x="date",
                y="value",
                label=label_obs,
                s=5,
                lw=0,
                color="black",
                zorder=2,
                ax=axs[j, i],
            )
            sns.lineplot(
                data=fitted_loc,
                x="date",
                y="mean",
                label=label_fit,
                zorder=0,
                lw=2,
                alpha=0.5,
                ax=axs[j, i],
            )
            axs[j, i].fill_between(
                fitted_loc["date"],
                fitted_loc["lower"],
                fitted_loc["upper"],
                alpha=0.3,
                edgecolor="none",
            )
            if df_predicted is not None:
                sns.lineplot(
                    data=predicted_loc,
                    x="date",
                    y="mean",
                    label=label_pred,
                    zorder=3,
                    ax=axs[j, i],
                )
                axs[j, i].fill_between(
                    predicted_loc["date"],
                    predicted_loc["lower"],
                    predicted_loc["upper"],
                    alpha=0.3,
                    edgecolor="none",
                )
            axs[j, i].set_title(
                f"site: {locations.site_code[k]} , lat: {locations.lat[k]} , lon: {locations.lon[k]}"  # noqa: E501
            )
            axs[j, i].spines[["right", "top"]].set_visible(False)
            k += 1
        axs[j, 0].set_ylabel(f"{gas.upper()} in {unit}")
    axs[0, 0].legend(frameon=False, handlelength=0.5)

    return fig, axs


def plot_prophet_components(
    components: dict[str, pd.DataFrame],
    gas: str,
    unit: str,
    x_years: tuple[int, int],
    figsize: Optional[tuple[int, int]] = None,
) -> Any:
    """Plot trend and seasonality components from Prophet models.

    Creates a figure with two rows: trend components (top) and
    seasonality components (bottom), with separate panels for
    each region (Southern, Tropical, Northern).

    Parameters
    ----------
    components
        Dictionary with keys 'southern', 'tropical', 'northern',
        each containing a DataFrame with columns 'ds', 'trend', 'yearly'.
    gas
        Name of greenhouse gas (e.g., 'co2', 'ch4').
    unit
        Unit of measurement (e.g., 'ppm', 'ppb').

    x_years
        Tuple of start and end years for the x-axis.
    figsize
        Size of the figure. If None, uses (12, 8).

    Returns
    -------
    Any
        fig, axs tuple from matplotlib.
    """
    if figsize is None:
        figsize = (12, 8)

    fig, axs = plt.subplots(
        2, 3, figsize=figsize, constrained_layout=True, sharex="col"
    )

    region_order = ["southern", "tropical", "northern"]
    region_labels = ["Southern", "Tropical", "Northern"]
    colors = ["tab:orange", "tab:green", "tab:blue"]

    # Plot trend components (top row)
    for i, (region, label, color) in enumerate(
        zip(region_order, region_labels, colors)
    ):
        if region in components:
            df = components[region]
            axs[0, i].plot(df["ds"], df["trend"], color=color, lw=2)
            axs[0, i].set_title(f"{label} Hemisphere")
            axs[0, i].set_ylabel(f"Trend [{unit}]")
            axs[0, i].spines[["right", "top"]].set_visible(False)
            axs[0, i].grid(True, alpha=0.3)

    # Plot seasonality components (bottom row)
    for i, (region, label, color) in enumerate(
        zip(region_order, region_labels, colors)
    ):
        if region in components:
            df = components[region]
            axs[1, i].plot(df["ds"], df["yearly"], color=color, lw=2)
            axs[1, i].set_title(f"{label} Hemisphere")
            axs[1, i].set_ylabel(f"Seasonality [{unit}]")
            axs[1, i].set_xlabel("Date")
            axs[1, i].spines[["right", "top"]].set_visible(False)
            axs[1, i].grid(True, alpha=0.3)
            axs[1, i].axhline(y=0, color="black", linestyle="--", lw=0.5, alpha=0.5)

    fig.suptitle(
        f"{gas.upper()} Prophet Model Components: Trend and Seasonality",
        fontsize=14,
        y=1.02,
    )

    xlim_start = pd.Timestamp(f"{x_years[0]}-01-01")
    xlim_end = pd.Timestamp(f"{x_years[1]}-12-31")

    tick_years = [
        pd.Timestamp(f"{year}-01-01") for year in range(x_years[0], x_years[1], 5)
    ]

    for row in axs:
        for ax in row:
            ax.set_xlim(xlim_start, xlim_end)
            ax.set_xticks(tick_years)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Show only year
            ax.tick_params(axis="x", labelsize=8)  # Make x-tick labels smaller

    if fig._suptitle is not None:
        current_text = fig._suptitle.get_text()
        fig.suptitle(current_text, fontsize=14, y=1.08)

    return fig, axs
