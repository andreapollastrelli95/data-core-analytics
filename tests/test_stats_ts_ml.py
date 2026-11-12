import numpy as np
import pandas as pd
import pytest

import ml_engine
import stats_engine as st
import ts_engine as ts
from cleaner import DataError


def test_describe_matches_pandas(sales):
    num = st.describe_numeric(sales).set_index("colonna")
    assert num.loc["Fatturato", "media"] == pytest.approx(sales["Fatturato"].mean())
    assert num.loc["Fatturato", "mediana"] == pytest.approx(sales["Fatturato"].median())
    cat = st.describe_categorical(sales).set_index("colonna")
    assert cat.loc["Canale", "distinti"] == 2


def test_correlation(sales):
    corr, pairs = st.correlation(sales, "pearson")
    assert corr.loc["Fatturato", "Fatturato"] == pytest.approx(1.0)
    top = pairs.iloc[0]
    assert {top["variabile A"], top["variabile B"]} == {"Prezzo Unitario", "Fatturato"}
    with pytest.raises(DataError):
        st.correlation(sales[["Regione", "Fatturato"]])


def test_outliers_detect_remove_clip():
    df = pd.DataFrame({"x": [10.0] * 30 + [11.0] * 30 + [1000.0]})
    table, mask = st.detect_outliers(df, ["x"], "iqr", 1.5)
    assert int(table["outlier"][0]) == 1 and int(mask.sum()) == 1
    assert len(st.remove_outliers(df, ["x"], "iqr", 1.5).df) == 60
    assert st.clip_outliers(df, ["x"], "iqr", 1.5).df["x"].max() < 1000


@pytest.mark.parametrize("test,v1,v2", [
    ("normality", "Fatturato", None), ("ttest", "Fatturato", "Canale"),
    ("anova", "Fatturato", "Regione"), ("chi2", "Regione", "Canale"),
    ("corr", "Quantita", "Fatturato")])
def test_hypothesis_tests_run(sales, test, v1, v2):
    text = st.run_test(sales, test, v1, v2)
    assert "p-value" in text or "p =" in text
    assert st.test_code(test, v1, v2, 0.05)


def test_hypothesis_test_validation(sales):
    with pytest.raises(DataError):
        st.run_test(sales, "ttest", "Fatturato", None)
    with pytest.raises(DataError):
        st.run_test(sales, "corr", "Fatturato", "Fatturato")


def test_series_total_is_preserved(sales):
    series, info = ts.build_series(sales, "Data Ordine", "Fatturato", "MS", "sum")
    assert series.sum() == pytest.approx(sales["Fatturato"].sum())
    assert info["periods"] == 12 and "Data Ordine" in ts.datetime_candidates(sales)


@pytest.mark.parametrize("method", ["ma", "ses", "holt"])
def test_forecast_methods(sales, method):
    series, _ = ts.build_series(sales, "Data Ordine", "Fatturato", "MS", "sum")
    res = ts.forecast(series, method, 3)
    fc = res["forecast"]
    assert len(fc) == 3 and fc["previsione"].notna().all()
    assert (fc["limite_inferiore_95"] <= fc["previsione"]).all()
    assert res["metrics"]["MAE"] >= 0


def test_forecast_needs_history():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2025-01-01", periods=3, freq="MS"))
    with pytest.raises(DataError):
        ts.forecast(short, "holt", 2)


def test_regression_recovers_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "gruppo": rng.choice(["a", "b"], 300), "y": 3 * x + 1})
    res = ml_engine.train(df, "y", ["x", "gruppo"], "regression", "linear")
    assert res.metrics["R²"] > 0.99
    assert "y" in ml_engine.training_code(res)
    assert "pred" in " ".join(map(str, ml_engine.add_predictions(df, res).df.columns)).lower()


def test_classification(sales):
    res = ml_engine.train(sales, "Canale", ["Quantita", "Fatturato", "Regione"], "classification", "rf",
                          params={"n_estimators": 20})
    assert 0 <= res.metrics["Accuracy"] <= 1 and res.confusion is not None


def test_train_validation(sales):
    with pytest.raises(DataError):
        ml_engine.train(sales, "Fatturato", [], "regression", "linear")
