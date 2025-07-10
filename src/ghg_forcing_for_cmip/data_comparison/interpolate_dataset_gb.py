"""
Interpolate binned dataset to 5 by 5 grid
"""

from __future__ import annotations

from typing import cast

import cftime  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from prefect import flow, task
from scipy.interpolate import griddata  # type: ignore

from ghg_forcing_for_cmip.data_comparison import CONFIG


@task(
    name="get_around_the_world_grid",
    description="Get the grid required for 'round the world' interpolation",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def get_round_the_world_grid(
    inv: npt.NDArray[np.float64], is_lon: bool = False
) -> npt.NDArray[np.float64]:
    """
    Get the grid required for 'round the world' interpolation

    Parameters
    ----------
    inv
        Input values

    is_lon
        Whether the input values represent longitudes or not

    Returns
    -------
        Grid to use for 'round the world' interpolation
    """
    if is_lon:
        out = np.hstack([inv - 360, inv, inv + 360])

    else:
        out = np.hstack([inv, inv, inv])

    return out


@task(
    name="interpolate_binned_data",
    description="Interpolate binned data",
    cache_policy=CONFIG.CACHE_POLICIES,
    refresh_cache=True,
)
def interpolate(
    ymdf: pd.DataFrame, value_column: str = "value"
) -> npt.NDArray[np.float64]:
    """
    Interpolate binned values

    Uses 'round the world' interpolation,
    i.e. longitudes interpolate based on values in both directions.

    Parameters
    ----------
    ymdf
        :obj:`pd.DataFrame` on which to do the interpolation.

    value_column
        The column in ``ymdf`` which contains the values in each bin.

    Returns
    -------
        Interpolated values in each bin, derived from ``ymdf``.
    """
    # Have to be hard-coded to ensure ordering is correct
    spatial_bin_columns = list(("lon", "lat"))

    missing_spatial_cols = [c for c in spatial_bin_columns if c not in ymdf.columns]
    if missing_spatial_cols:
        msg = f"{missing_spatial_cols=}"
        raise AssertionError(msg)

    ymdf_spatial_points = ymdf[spatial_bin_columns].to_numpy()

    lon_grid, lat_grid = np.meshgrid(CONFIG.LON_BIN_CENTRES, CONFIG.LAT_BIN_CENTRES)
    # Malte's trick, duplicate the grids so we can go
    # 'round the world' with interpolation
    lon_grid_interp = get_round_the_world_grid(lon_grid, is_lon=True)
    lat_grid_interp = get_round_the_world_grid(lat_grid)

    points_shift_back = ymdf_spatial_points.copy()
    points_shift_back[:, 0] -= 360
    points_shift_forward = ymdf_spatial_points.copy()
    points_shift_forward[:, 0] += 360
    points_interp = np.vstack(
        [
            points_shift_back,
            ymdf_spatial_points,
            points_shift_forward,
        ]
    )
    values_interp = get_round_the_world_grid(ymdf[value_column].to_numpy())

    res_linear_interp = griddata(
        points=points_interp,
        values=values_interp,
        xi=(lon_grid_interp, lat_grid_interp),
        method="linear",
    )
    res_linear = res_linear_interp[:, lon_grid.shape[1] : lon_grid.shape[1] * 2]

    # Have to return the transpose to ensure we match the column order specified above.
    # This is super flaky, alter with care!
    return cast(npt.NDArray[np.float64], res_linear.T)


@task(
    name="convert_to_da_xarray",
    description="Create a xr.DataArray object from interpolated values",
    cache_policy=CONFIG.CACHE_POLICIES,
)
def to_xarray_dataarray(
    bin_averages_df: pd.DataFrame,
    data: list[npt.NDArray[np.float64]],
    times: list[cftime.datetime],
    name: str,
) -> xr.DataArray:
    """
    Create an :obj:`xr.DataArray` from interpolated values

    Parameters
    ----------
    bin_averages_df
        Initial bin averages :obj:`pd.DataFrame`.
        This is just used to extract metadata, e.g. units.

    data
        Data for the array. We assume that this has already been interpolated
        onto a lat, lon grid defined by {py:const}`LAT_BIN_CENTRES`,
        {py:const}`LON_BIN_CENTRES` and {py:func}`get_spatial_dimension_order`.

    times
        Time axis of the data

    name
        Name of the output :obj:`xr.DataArray`

    Returns
    -------
        Created :obj:`xr.DataArray`
    """
    # lat and lon come from the module scope (not best pattern, not the worst)
    units = bin_averages_df["unit"].unique()
    if len(units) > 1:
        msg = f"Need unit conversion, {units=}"
        raise AssertionError(msg)

    unit = units[0]

    da = xr.DataArray(
        name=name,
        data=np.array(data),
        dims=["time", *tuple(v for v in ("lon", "lat"))],
        coords=dict(
            time=times,
            lon=CONFIG.LON_BIN_CENTRES,
            lat=CONFIG.LAT_BIN_CENTRES,
        ),
        attrs=dict(
            description="Interpolated spatial data",
            units=unit,
        ),
    )

    return da


@flow(name="interpolation_flow", description="interpolate binned data")
def interpolation_flow(
    path_to_csv: str,
    gas: str,
) -> None:
    """
    Run interpolation flow on binned data

    Parameters
    ----------
    path_to_csv :
        path where binned data is stored

    gas :
        target greenhouse gas variable
    """
    times_l = []
    interpolated_dat_l = []

    d_binned_avg = pd.read_csv(path_to_csv + f"/{gas}/{gas}_binned.csv")

    # required for merging later (otherwise time has type object)
    d_binned_avg["time"] = pd.to_datetime(d_binned_avg.time.astype(str), utc=True)

    for (time, year, month), ymdf in d_binned_avg.groupby(["time", "year", "month"]):
        if ymdf.shape[0] < CONFIG.MIN_POINTS_FOR_SPATIAL_INTERPOLATION:
            msg = (
                f"Not enough data ({ymdf.shape[0]} data points) for {time=}, "
                f"not performing spatial interpolation"
            )
            print(msg)
            continue

        ymdf.dropna(inplace=True)

        interpolated_ym = interpolate(ymdf)

        if np.isnan(interpolated_ym).any():
            msg = (
                f"Nan data after interpolation for {time=}, "
                f"not including spatial interpolation in output"
            )
            print(msg)

        # else:
        interpolated_dat_l.append(interpolated_ym)
        times_l.append(pd.to_datetime(time, utc=True))

    out = to_xarray_dataarray(
        name="value",
        bin_averages_df=d_binned_avg,
        data=interpolated_dat_l,
        times=times_l,
    )

    df_out = out.to_dataframe().reset_index()
    df_out = df_out.merge(
        d_binned_avg.drop(columns="value"), on=["time", "lon", "lat"], how="outer"
    )

    time_frac = d_binned_avg[["time", "time_fractional"]].drop_duplicates()
    df_out["time_fractional"] = df_out["time"].map(
        time_frac.set_index("time")["time_fractional"]
    )

    df_out["month"] = df_out.time.dt.month
    df_out["year"] = df_out.time.dt.year
    df_out["gas"] = gas
    df_out["unit"] = np.where(gas == "ch4", "ppb", "ppm")

    # remove NAN from dataframe and drop duplicates
    df_out.dropna(subset="value", inplace=True)
    df_out.drop_duplicates(inplace=True)

    df_out.to_csv(path_to_csv + f"/{gas}/{gas}_interpolated.csv")


if __name__ == "__main__":
    interpolation_flow(path_to_csv="data/downloads", gas="ch4")
