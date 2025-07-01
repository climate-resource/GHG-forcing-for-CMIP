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
    d2 = d_vertical[["time_fractional", "lat", "lon", "value_orig"]].reset_index(
        drop=True
    )
    d2.rename(columns={"value_orig": "value"}, inplace=True)

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
