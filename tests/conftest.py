import numpy as np
import pandas as pd
import pytest

from w37_reconciliation.pipeline import load_contract


@pytest.fixture(scope="session")
def contract() -> dict:
    return load_contract()


@pytest.fixture
def S_small() -> np.ndarray:
    """1 total + 3 feuilles."""
    return np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


@pytest.fixture
def S_df_small(S_small) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": ["T", "a", "b", "c"],
                         "a": S_small[:, 0], "b": S_small[:, 1], "c": S_small[:, 2]})


@pytest.fixture
def coherent(S_small):
    """Fabrique de prévisions cohérentes (h × 4)."""
    def make(h: int = 5, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).uniform(10, 50, size=(h, 3)) @ S_small.T
    return make
