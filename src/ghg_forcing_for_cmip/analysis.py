"""
analysis pipeline for incorporating EO into GB
"""

from typing import Any

import bambi as bmb
import pandas as pd


def fit_gb_from_eo(
    formula: str, df: pd.DataFrame, seed: int, categorical: list[str]
) -> Any:
    """
    Fit Bayesian regression model to predict GB from EO

    Parameters
    ----------
    formula :
        statistical model formulation

    df :
        data used for fitting

    seed :
        seed for reproducibility

    categorical :
        list with variables in formula considered
        to be categorical

    Returns
    -------
    Any
        model and inference object
    """
    model = bmb.Model(formula=formula, data=df, categorical=categorical)
    idata = model.fit(draws=1000, chains=4, random_seed=seed, target_accept=0.95)

    return model, idata
