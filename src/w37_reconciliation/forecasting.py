"""Prévisions de base indépendantes (une par série) et résidus in-sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from .data import to_wide


def fit_base_forecasts(train: pd.DataFrame, h: int, season: int = 4,
                       freq: str = "QS") -> tuple[pd.DataFrame, pd.DataFrame]:
    """AutoETS + SeasonalNaive par série.

    Retourne (Y_hat, Y_fit) : prévisions hors échantillon et valeurs ajustées in-sample.
    `fitted=True` est indispensable : sans valeurs ajustées, pas de résidus, donc pas de W pour mint_shrink.
    """
    sf = StatsForecast(models=[AutoETS(season_length=season), SeasonalNaive(season_length=season)],
                       freq=freq, n_jobs=1)
    Y_hat = sf.forecast(df=train, h=h, fitted=True)
    return Y_hat, sf.forecast_fitted_values()


def residual_matrix(Y_fit: pd.DataFrame, order: list[str], model: str = "AutoETS") -> np.ndarray:
    """Résidus one-step in-sample E (T × m), lignes incomplètes retirées."""
    return (to_wide(Y_fit, order, "y") - to_wide(Y_fit, order, model)).dropna().to_numpy()
