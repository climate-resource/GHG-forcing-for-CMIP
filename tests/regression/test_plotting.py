"""
Regression tests for plotting.py module
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from ghg_forcing_for_cmip.plotting import (
    plot_average_hemisphere,
    plot_collocated_rmse,
    plot_map,
    plot_monthly_average,
)

pytest_regressions = pytest.importorskip("pytest_regressions")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(scope="module")
def df_test1():
    return pd.read_csv("tests/test_data/plotting/df_test_plot.csv")


@pytest.fixture(scope="module")
def df_test2():
    return pd.read_csv("tests/test_data/plotting/co2_collocated.csv")


@pytest.fixture(scope="module")
def df_test3():
    return pd.read_csv("tests/test_data/plotting/ch4_collocated.csv")


os.makedirs("tests/regression/test_plotting", exist_ok=True)

pytest.importorskip("geopandas")
pytest.importorskip("geodatasets")


def test_plot_map(image_regression, df_test1):
    fig, axs = plt.subplots()
    axs = plot_map(df_test1, title="test", axs=axs)

    out_file = "tests/regression/test_plotting/fig1.png"
    fig.savefig(out_file, pad_inches=0)

    image_regression.check(image_data=Path(out_file).read_bytes(), diff_threshold=0.01)


def test_plot_monthly_average(image_regression, df_test2):
    fig, axs = plt.subplots()
    axs = plot_monthly_average(df_test2, "$CO_2$", axs)

    out_file = "tests/regression/test_plotting/fig2.png"
    fig.savefig(out_file, pad_inches=0)

    image_regression.check(image_data=Path(out_file).read_bytes(), diff_threshold=0.01)


def test_plot_average_hemisphere(image_regression, df_test2):
    plot_average_hemisphere(df_test2, "$CO_2$")

    out_file = "tests/regression/test_plotting/fig3.png"
    plt.savefig(out_file, pad_inches=0)

    image_regression.check(image_data=Path(out_file).read_bytes(), diff_threshold=0.01)


def test_plot_collocated_rmse(image_regression, df_test2, df_test3):
    plot_collocated_rmse(df_test2, df_test3, "rmse")

    out_file = "tests/regression/test_plotting/fig4.png"
    plt.savefig(out_file, pad_inches=0)

    image_regression.check(image_data=Path(out_file).read_bytes(), diff_threshold=0.01)
