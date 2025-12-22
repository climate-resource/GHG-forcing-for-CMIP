"""
Plotting functions for tutorials in documentation

"""

from typing import Any, Optional

import geodatasets
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    figsize: tuple[int, int] = (7, 3),
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
        import geopandas  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="geopandas"
        ) from exc

    try:
        from geodatasets import get_path  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plotting", requirement="geodatasets"
        ) from exc

    gdf = geopandas.GeoDataFrame(
        d,
        geometry=geopandas.points_from_xy(d[lon_value], d[lat_value]),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(get_path("naturalearth.land"))
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
        import seaborn as sns  # type: ignore[import-untyped]
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
        data=df,
        x="date",
        y="value_gb",
        hue="hemisphere",
        ax=axs,
        alpha=0.2,
        legend=None,
    )
    sns.lineplot(data=df, x="date", y="value_gb_pred", hue="hemisphere", ax=axs)
    axs.set_ylabel(f"{gas.upper()} in {unit}")
    axs.spines[["right", "top"]].set_visible(False)
    axs.legend(frameon=False, handlelength=0.5, ncol=3)

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
