import pandas as pd
import pytest

import sql_engine
import transform_engine as tr
from cleaner import DataError


def test_group_by_totals_match(sales):
    res = tr.group_by(sales, ["Regione"], [("Fatturato", "sum"), ("Quantita", "mean")])
    assert res.df["Fatturato_sum"].sum() == pytest.approx(sales["Fatturato"].sum())
    assert res.df["n_righe"].sum() == len(sales)
    assert len(res.df) == 5                      # 4 regioni + il gruppo senza regione


def test_group_by_validation(sales):
    with pytest.raises(DataError):
        tr.group_by(sales, [], [("Fatturato", "sum")])
    with pytest.raises(DataError):
        tr.group_by(sales, ["Regione"], [("Canale", "sum")])


def test_merge_left_keeps_rows(small):
    right = pd.DataFrame({"cliente": ["A", "B", "C"], "citta": ["Roma", "Milano", "Bari"]})
    res = tr.merge(small, right, "left", ["cliente"], ["cliente"])
    assert len(res.df) == len(small)
    assert res.df.loc[res.df["cliente"] == "B", "citta"].eq("Milano").all()


def test_merge_rejects_mismatched_keys(small):
    right = pd.DataFrame({"cliente": [1, 2]})
    with pytest.raises(DataError):
        tr.merge(small, right, "inner", ["cliente"], ["cliente"])


def test_concat(small):
    assert len(tr.concat(small, small).df) == 2 * len(small)


def test_sql_query_and_code(sales):
    q = 'SELECT Canale, SUM(Fatturato) AS f FROM df GROUP BY Canale ORDER BY f DESC'
    out = sql_engine.run_query(q, {"df": sales})
    assert out["f"].sum() == pytest.approx(sales["Fatturato"].sum())
    assert "duckdb.connect" in sql_engine.query_code(q, ["df"])


def test_sql_errors(sales):
    with pytest.raises(DataError):
        sql_engine.run_query("", {"df": sales})
    with pytest.raises(DataError):
        sql_engine.run_query("SELECT colonna_inesistente FROM df", {"df": sales})


@pytest.mark.parametrize("query", [
    "SELECT * FROM read_csv('{path}')",
    "SELECT * FROM '{path}'",
    "COPY df TO '{out}'",
    "ATTACH '{out}.db' AS altro",
    "INSTALL httpfs",
])
def test_sql_sandbox_blocks_files(sales, tmp_path, query):
    """Le query scritte dall'IA non possono leggere né scrivere file del computer."""
    from conftest import SAMPLE

    out = tmp_path / "rubato.csv"
    with pytest.raises(DataError):
        sql_engine.run_query(query.format(path=SAMPLE, out=out), {"df": sales}, sandbox=True)
    assert not out.exists()


def test_sql_sandbox_still_reads_dataframes(sales):
    out = sql_engine.run_query("SELECT COUNT(*) AS n FROM df", {"df": sales}, sandbox=True)
    assert int(out["n"][0]) == len(sales)
