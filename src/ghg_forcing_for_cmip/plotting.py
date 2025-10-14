"""
Plotting functions for tutorials in documentation

"""

from typing import Any

import numpy as np
import pandas as pd

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
) -> None:
    """
    Plot world map with observation stations

    Parameters
    ----------
    d :
        dataframe with ghg concentrations and lat, lon information

    title :
        title of the plot
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
    gdf.plot(figsize=(7, 3), ax=axs, color="red", marker=marker, markersize=markersize)
    axs.set_title(title)


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


def plot_vertical_apriori(d: pd.DataFrame, axs: Any, gas: str) -> None:
    """
    Plot vertical profile of apriori concentration in OBS4MIPS

    Parameters
    ----------
    d :
        dataframe incl. vertical information

    axs :
        axes from plt.subplots object

    gas :
        either "ch4" or "co2"
    """
    d = (
        d.groupby("pre")
        .agg({"vmr_profile_apriori": "mean", "value_eo": "mean"})
        .reset_index()
    )
    axs.plot(d.vmr_profile_apriori, 1 - d.pre, label="apriori-profile")
    axs.plot(d.value_eo, 1 - d.pre, label="column-average")
    axs.set_xlabel(gas)
    axs.set_yticks(np.round(d.pre.unique(), 2))
    axs.set_yticklabels(np.round(d.pre.unique(), 2)[::-1])
    axs.legend(frameon=False)


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

    _, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(5, 7))
    sns.scatterplot(
        data=df_measure.sort_values(by=f"{measure}_co2"),
        x=f"{measure}_co2",
        y="site_code",
        ax=axs[0],
    )
    sns.scatterplot(
        data=df_measure.sort_values(by=f"{measure}_ch4"),
        x=f"{measure}_ch4",
        y="site_code",
        ax=axs[1],
    )
    for i in range(2):
        axs[i].tick_params(axis="both", labelsize=8)
        axs[i].set_xlabel(measure.upper())
    axs[1].set_ylabel("")
    axs[1].set_title(
        (
            f"CH4-GB vs. CH4-EO\n avg. {measure.upper()}:"
            + f"{df_measure[f'{measure}_ch4'].mean():.2f}"
        ),
        fontsize="medium",
    )
    axs[0].set_title(
        (
            f"CO2-GB vs. CO2-EO\n avg. {measure.upper()}:"
            + f"{df_measure[f'{measure}_co2'].mean():.2f}"
        ),
        fontsize="medium",
    )
    return axs
