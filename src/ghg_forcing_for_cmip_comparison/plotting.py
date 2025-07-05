"""
Plotting helpers for inspecting datasets
"""

from typing import Optional

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geodatasets import get_path

from ghg_forcing_for_cmip_comparison.utils import compute_weighted_avg


def plot_map(d: pd.DataFrame, title: str) -> None:
    """
    Plot world map with observation stations

    Parameters
    ----------
    d :
        dataframe with ghg concentrations and lat, lon information

    title :
        title of the plot
    """
    gdf = geopandas.GeoDataFrame(
        d, geometry=geopandas.points_from_xy(d.longitude, d.latitude), crs="EPSG:4326"
    )

    world = geopandas.read_file(get_path("naturalearth.land"))
    ax = world.plot(color="white", edgecolor="grey")
    gdf.plot(figsize=(7, 3), ax=ax, color="red", marker="x", markersize=20)
    plt.title(title)
    plt.show()


def plot_map_grid(d_binned: pd.DataFrame) -> None:
    """
    Plot earth map with grid lines for each bin

    Parameters
    ----------
    d_binned :
        binned dataset
    """
    gdf = geopandas.GeoDataFrame(
        d_binned,
        geometry=geopandas.points_from_xy(d_binned.lon, d_binned.lat),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(get_path("naturalearth.land"))
    ax = world.plot(color="white", edgecolor="grey")

    [ax.axhline(lat, lw=0.5) for lat in d_binned.lat.unique()]
    [ax.axvline(lon, lw=0.5) for lon in d_binned.lon.unique()]

    gdf.plot(figsize=(7, 3), ax=ax, color="red", marker="x", markersize=20)
    plt.title("Binning observations on a 5° x 5° grid")
    plt.show()


def plot_map_interpolated(  # noqa: PLR0913
    d_interpol: pd.DataFrame,
    d_binned: pd.DataFrame,
    year: int,
    month: int,
    gas: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot world map with interpolated ghg concentrations and lat, lon info

    Parameters
    ----------
    d_interpol :
        dataframe with interpolated ghg concentrations

    d_binned :
        dataframe with binned ghg concentrations

    year :
        year of interest

    month :
        month of interest

    gas :
        target greenhouse gas variable

    vmin :
        minimum ghg concentration value used for legend

    vmax :
        maximum ghg concentration value used for legend
    """
    d_filtered = d_interpol[(d_interpol.year == year) & (d_interpol.month == month)]
    d_filtered2 = d_binned[(d_binned.year == year) & (d_binned.month == month)]

    gdf = geopandas.GeoDataFrame(
        d_filtered,
        geometry=geopandas.points_from_xy(d_filtered.lon, d_filtered.lat),
        crs="EPSG:4326",
    )

    gdf2 = geopandas.GeoDataFrame(
        d_filtered2,
        geometry=geopandas.points_from_xy(d_filtered2.lon, d_filtered2.lat),
        crs="EPSG:4326",
    )

    world = geopandas.read_file(get_path("naturalearth.land"))
    ax = world.plot(color="white", edgecolor="grey")

    gdf.plot(
        column="value",
        figsize=(7, 3),
        ax=ax,
        alpha=0.3,
        markersize=10,
        marker="s",
        legend=True,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            "label": f"{gas} concentration in {np.where(gas == 'ch4', 'ppb', 'ppm')}",
            "orientation": "horizontal",
        },
    )
    gdf2.plot(
        figsize=(7, 3),
        ax=ax,
        markersize=15,
        marker="x",
        color="black",
        label="observed",
    )
    plt.title(f"Interpolated data for {year}-{month}")
    plt.legend(fontsize="x-small")
    plt.show()


def plot_annual_concentration(d: pd.DataFrame, gas: str) -> None:
    """
    Plot global, annual ghg concentration

    Together with global, annual conc. of northern and
    southern hemisphere

    Parameters
    ----------
    d :
        dataframe with interpolated ghg concentrations

    gas :
        target greenhouse gas variable
    """
    d_lat = compute_weighted_avg(d, ["time_fractional", "lat"])
    d_global = compute_weighted_avg(d, ["time_fractional"])

    _ = plt.figure(figsize=(7, 4))
    sns.lineplot(
        x="time_fractional",
        y="value",
        data=d_lat[d_lat.lat > 0].groupby(["time_fractional"]).agg({"value": "mean"}),
        label="northern hemisphere",
    )
    sns.lineplot(
        x="time_fractional",
        y="value",
        data=d_lat[d_lat.lat < 0].groupby(["time_fractional"]).agg({"value": "mean"}),
        label="southern hemisphere",
    )
    sns.lineplot(x="time_fractional", y="value", data=d_global, label="global")
    plt.ylabel(f"{gas.upper()} in {np.where(gas=='ch4', 'ppb', 'ppm')}")
    plt.xlabel("year/month")
    plt.show()


def plot_vertical(
    d_vertical: pd.DataFrame,
    gas: str,
    lat: float = 2.5,
    p_surface: int | float = 950,
    p_top: int | float = 50,
) -> None:
    """
    Plot vertical dimension of ground-based (interpolated data)

    Parameters
    ----------
    d_vertical:
        dataframe with vertical ghg concentration

    gas :
        target greenhouse gas variable

    lat :
        selected latitude

    p_surface :
        surface pressure level

    p_top :
        atmospheric top pressure level
    """
    d2 = d_vertical[
        ["time_fractional", "lat", "lon", "value_without_vertical"]
    ].reset_index(drop=True)
    d2.rename(columns={"value_without_vertical": "value"}, inplace=True)

    d1 = compute_weighted_avg(d_vertical, ["time_fractional", "lat", "pre"])
    d2 = compute_weighted_avg(d2, ["time_fractional", "lat"])

    _, axs = plt.subplots(1, 1, figsize=(7, 4))
    sns.lineplot(
        data=d1[d1.lat == lat],
        x="time_fractional",
        y="value",
        estimator=None,
        alpha=0.2,
        errorbar=("pi", 100),
        ax=axs,
    )
    sns.lineplot(
        data=d1[(d1.lat == lat) & (d1.pre == p_surface)],
        x="time_fractional",
        y="value",
        ax=axs,
        label="surface",
    )
    sns.lineplot(
        data=d1[(d1.lat == lat) & (d1.pre == p_top)],
        x="time_fractional",
        y="value",
        ax=axs,
        label="top",
    )
    sns.lineplot(
        data=d2[d2.lat == lat],
        x="time_fractional",
        y="value",
        ax=axs,
        linestyle="dotted",
        color="black",
        label="original",
    )
    plt.legend(title="pressure")
    plt.title(f"Range of {gas.upper()} concentration across vertical dimension")
    plt.ylabel(f"{gas.upper()} in {np.where(gas == 'ch4', 'ppb', 'ppm')}")
    plt.xlabel("year/month")
    plt.show()


def plot_annual_concentration_comparison(
    d_vertical: pd.DataFrame, d_interpol: pd.DataFrame, gas: str
) -> None:
    """
    Plot monthly, ghg-conc. for northern/southern hemis. and global mean

    Parameters
    ----------
    d_vertical:
        dataset with ground-based data incl. modelled vertical distribution

    d_interpol:
        dataset with interpolated ground-based data points to 5 by 5 grid

    gas:
        target greenhouse gas variable
    """
    _, axs = plt.subplots(1, 1, figsize=(7, 4))

    for d, line, alpha, lw, label in zip(
        [d_vertical, d_interpol], ["solid", "dashed"], [1.0, 0.5], [2, 1], [True, False]
    ):
        d_lat = compute_weighted_avg(d, ["time_fractional", "lat"])
        d_global = compute_weighted_avg(d, ["time_fractional"])

        sns.lineplot(
            x="time_fractional",
            y="value",
            data=d_lat[d_lat.lat > 0.0]
            .groupby(["time_fractional"])
            .agg({"value": "mean"}),
            label=np.where(label, "northern hemisphere", ""),
            linestyle=line,
            color="grey",
            alpha=alpha,
            linewidth=lw,
            ax=axs,
        )
        sns.lineplot(
            x="time_fractional",
            y="value",
            data=d_lat[d_lat.lat < 0.0]
            .groupby(["time_fractional"])
            .agg({"value": "mean"}),
            label=np.where(label, "southern hemisphere", ""),
            linestyle=line,
            color="green",
            alpha=alpha,
            linewidth=lw,
            ax=axs,
        )
        sns.lineplot(
            x="time_fractional",
            y="value",
            data=d_global,
            label=np.where(label, "global", ""),
            linestyle=line,
            color="red",
            alpha=alpha,
            linewidth=lw,
            ax=axs,
        )
    plt.ylabel(f"{gas.upper()} in {np.where(gas=='ch4', 'ppb', 'ppm')}")
    plt.xlabel("year/month")
    plt.title(
        "Comparison between GHG-concentration \n "
        + "with (solid) & without (dashed) vertical dimension"
    )
    plt.show()


def plot_map_combined(d: pd.DataFrame, years: list[int], month: int, gas: str) -> None:
    """
    Plot map with ghg concentrations for specific year and month

    Parameters
    ----------
    d :
        dataset with ghg-concentration data in column "value"

    years :
        list of years for which plot should be created

    month :
        for which month plot should be created

    gas :
        target greenhouse gas variable
    """
    _, axs = plt.subplots(1, int(len(years)), constrained_layout=True, figsize=(15, 5))
    for i, year in enumerate(years):
        d_filtered = d[(d.year == year) & (d.month == month)]

        gdf = geopandas.GeoDataFrame(
            d_filtered,
            geometry=geopandas.points_from_xy(d_filtered.lon, d_filtered.lat),
            crs="EPSG:4326",
        )

        world = geopandas.read_file(get_path("naturalearth.land"))
        ax = world.plot(color="white", edgecolor="grey", ax=axs[i])

        gdf.plot(
            column="value",
            figsize=(7, 3),
            ax=ax,
            alpha=0.3,
            markersize=10,
            marker="s",
            legend=True,
            cmap="seismic",
            legend_kwds={
                "label": f"{gas} concentration in {np.where(gas=='ch4', 'ppb', 'ppm')}",
                "orientation": "horizontal",
            },
        )
        axs[i].set_title(f"{year}-{month}")
    plt.show()


def plot_eo_gb_seasonal_annual(
    d_gb_no_AK: pd.DataFrame,
    d_gb_AK: pd.DataFrame,
    d_eo: pd.DataFrame,
    gas: str = "ch4",
) -> None:
    """
    Plot monthly, avg. ghg concentration satellite vs. ground-based

    Parameters
    ----------
    d_gb_no_AK :
        dataset for comparison of satellite data and ground-based data
        value column is ground-based data without AK

    d_gb_AK :
        dataset for comparison of satellite data and ground-based data
        value column is ground-based data with AK

    d_eo :
        dataset for comparison of satellite data and ground-based data
        value column is satellite data

    gas :
        target greenhouse gas variable
    """
    _, axs = plt.subplots(
        3, 2, figsize=(7, 4), sharex=True, sharey=True, constrained_layout=True
    )
    for j, (d_gb, name_gb) in enumerate(
        zip([d_gb_no_AK, d_gb_AK], ["ground-based no AK", "ground-based with AK"])
    ):
        for d, col, alpha, lw, label in zip(
            [d_gb, d_eo],
            ["red", "green"],
            [1.0, 1.0],
            [2, 2],
            [name_gb, "satellite"],
        ):
            d_lat = compute_weighted_avg(d, ["time_fractional", "lat"])
            d_global = compute_weighted_avg(d, ["time_fractional"])

            sns.lineplot(
                x="time_fractional",
                y="value",
                data=d_lat[d_lat.lat > 0.0]
                .groupby(["time_fractional"])
                .agg({"value": "mean"}),
                label=label,
                color=col,
                alpha=alpha,
                linewidth=lw,
                ax=axs[0, j],
            )
            sns.lineplot(
                x="time_fractional",
                y="value",
                data=d_lat[d_lat.lat < 0.0]
                .groupby(["time_fractional"])
                .agg({"value": "mean"}),
                color=col,
                alpha=alpha,
                linewidth=lw,
                ax=axs[2, j],
            )
            sns.lineplot(
                x="time_fractional",
                y="value",
                data=d_global,
                color=col,
                alpha=alpha,
                linewidth=lw,
                ax=axs[1, j],
            )
        axs[0, j].set_title("Northern Hemisphere")
        axs[2, j].set_title("Southern Hemisphere")
        axs[1, j].set_title("Global")
        [
            axs[i, j].set_ylabel(
                f"{gas.upper()} in {np.where(gas == 'ch4', 'ppb', 'ppm')}"
            )
            for i in range(3)
        ]
        axs[2, j].set_xlabel("year/month")
        axs[0, j].legend(frameon=False, fontsize="x-small", handlelength=0.3)
    plt.show()
