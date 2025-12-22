"""
analysis pipeline for incorporating EO into GB
"""

from typing import Any

import pandas as pd


def fit_gb_from_eo(formula: str, df: pd.DataFrame, seed: int) -> Any:
    """
    Fit Bayesian regression model to predict GB from EO

    Parameters
    ----------
    formula :
        statistical model formulation

    df :
        data used for fitting

    seed : int
        seed for reproducibility

    Returns
    -------
    Any
        model and inference object
    """
    pass
