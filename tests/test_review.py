import numpy as np
import pandas as pd

import ai_tools as T
import review_engine as rv


def values(text):
    return [c.values for c in rv.extract_numbers(text)]


def test_extract_italian_and_english_formats():
    assert values("Totale 1.234,56 euro")[0] == (1234.56,)
    assert values("Total 1,234.56")[0] == (1234.56,)
    assert set(values("circa 1.234 ordini")[0]) == {1234.0, 1.234}      # ambiguo: entrambe le letture
    assert values("quota 39,8%")[0] == (39.8,) and rv.extract_numbers("39,8%")[0].percent
    assert values("ricavi 53,7 mila")[0] == (53700.0,)
    assert values("p = 4.5e-3")[0] == (0.0045,)


def test_extract_ignores_dates_code_and_identifiers():
    assert values("Il 2025-03-01 e il 12/04/2025 alle 10:30") == []
    assert values("usa `df.head(20)` oppure la colonna p95 e Q1") == []
    assert values("```\nx = 12345\n```") == []


def test_grounding_direct_derived_and_missing():
    evidence = np.array([53691.52, 135016.34, 34218.19])
    g = rv.ground_numbers("Nord 53.691,52 su 135.016,34 (39,8%), +19.473,33 sul Centro; "
                          "margine 77.000,5", evidence)
    assert g["verificati"] == ["53.691,52", "135.016,34"]
    assert g["derivati"] == ["39,8%", "19.473,33"]
    assert g["non_trovati"] == ["77.000,5"]


def test_grounding_accepts_rounding_but_not_wrong_digits():
    evidence = np.array([53691.52])
    assert rv.ground_numbers("circa 53.692", evidence)["verificati"] == ["53.692"]
    assert rv.ground_numbers("circa 53,7 mila", evidence)["verificati"] == ["53,7 mila"]
    assert rv.ground_numbers("circa 53.700,10", evidence)["non_trovati"] == ["53.700,10"]


def test_grounding_fraction_as_percentage_and_user_numbers():
    g = rv.ground_numbers("quota 24,2% negli ultimi 90 giorni", np.array([0.242]),
                          known_text="considera gli ultimi 90 giorni")
    assert g == {"verificati": ["24,2%"], "derivati": [], "non_trovati": [], "totale": 1}


def test_small_integers_and_years_are_not_claims():
    assert rv.ground_numbers("Le 3 regioni nel 2025", np.empty(0))["totale"] == 0


def test_totals_check(sales):
    ctx = T.ToolContext(sales)
    res = T.run_tool("aggrega", {"per": ["Regione"],
                                 "metriche": [{"colonna": "Fatturato", "funzione": "sum"}]}, ctx)
    findings = rv.check_totals(res, sales)
    assert [f.level for f in findings] == [rv.OK, rv.OK, rv.WARN]       # 26 righe senza regione
    res.table.loc[0, "Fatturato_sum"] += 100                              # totale manomesso
    assert rv.ERROR in [f.level for f in rv.check_totals(res, sales)]


def test_join_many_to_many_is_flagged():
    orders = pd.DataFrame({"cliente": ["A", "A", "B"], "importo": [1, 2, 3]})
    many = pd.DataFrame({"cliente": ["A", "A", "B"], "agente": ["x", "y", "z"]})
    one = pd.DataFrame({"cliente": ["A", "B"], "citta": ["Roma", "Bari"]})
    q = "SELECT SUM(o.importo) FROM df o JOIN df2 c ON o.cliente = c.cliente"
    assert rv.check_join(q, T.ToolContext(orders, many))[0].level == rv.ERROR
    assert rv.check_join(q, T.ToolContext(orders, one))[0].level == rv.OK
    assert rv.check_join("SELECT * FROM df", T.ToolContext(orders, one)) == []
    assert rv.check_join("SELECT * FROM df NATURAL JOIN df2",
                         T.ToolContext(orders, one))[0].level == rv.WARN


def test_review_steps_reproducibility(sales):
    ctx = T.ToolContext(sales)
    args = {"query": "SELECT SUM(Fatturato) AS t FROM df"}
    good = T.run_tool("esegui_sql", args, ctx)
    findings = rv.review_steps([("sql", "esegui_sql", args, good)], ctx)
    assert findings[0].level == rv.OK and "1 passi su 1" in findings[0].title
    tampered = T.run_tool("esegui_sql", args, ctx)
    tampered.table = tampered.table * 2
    assert rv.review_steps([("sql", "esegui_sql", args, tampered)], ctx)[0].level == rv.ERROR
    assert rv.review_steps([], ctx)[0].level == rv.ERROR                 # risposta senza calcoli


def test_review_flags_truncated_and_empty_results(sales):
    ctx = T.ToolContext(sales)
    calls = []
    for q in ("SELECT * FROM df", "SELECT * FROM df WHERE Fatturato < 0"):
        calls.append((q, "esegui_sql", {"query": q}, T.run_tool("esegui_sql", {"query": q}, ctx)))
    titles = " ".join(f.title for f in rv.review_steps(calls, ctx))
    assert "troncato" in titles and "vuoto" in titles


def test_grounding_finding_levels():
    ok = {"verificati": ["1,5"], "derivati": [], "non_trovati": [], "totale": 1}
    bad = {"verificati": [], "derivati": [], "non_trovati": ["9,9"], "totale": 1}
    assert rv.grounding_finding(ok).level == rv.OK and rv.grounding_finding(bad).level == rv.ERROR
    assert rv.worst([rv.grounding_finding(ok), rv.grounding_finding(bad)]) == rv.ERROR
