"""
Plotting helpers for inspecting datasets
"""

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from geodatasets import get_path


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
