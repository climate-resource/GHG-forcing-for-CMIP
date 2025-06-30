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
