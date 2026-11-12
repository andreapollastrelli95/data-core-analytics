"""Configurazione comune dei test: percorso del progetto, cartella dati isolata, dataset di prova."""
import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cleaner  # noqa: E402

SAMPLE = os.path.join(ROOT, "esempi", "vendite_esempio.csv")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Dizionari e procedure dei test non toccano mai la cartella reale dell'utente."""
    monkeypatch.setenv("DCA_HOME", str(tmp_path / "dca_home"))


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    df, _code = cleaner.load_file(SAMPLE)
    return df


@pytest.fixture
def small() -> pd.DataFrame:
    return pd.DataFrame({
        "cliente": ["A", "A", "B", "B", "C", "C", "C", None],
        "canale": ["web", "web", "negozio", "web", "negozio", "web", "web", "web"],
        "importo": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 60.0, 5.0],
        "pezzi": [1, 2, 3, 4, 5, 6, 6, 1],
    })
