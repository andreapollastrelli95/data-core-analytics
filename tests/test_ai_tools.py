import json

import pytest

import ai_tools as T


@pytest.fixture
def ctx(sales):
    return T.ToolContext(sales)


def test_declarations_are_valid_json_schemas():
    decl = T.declarations()
    assert len(decl) == len(T.TOOLS) >= 9
    for d in decl:
        assert set(d) == {"name", "description", "parameters"}
        assert d["parameters"]["type"] == "object"
        assert set(d["parameters"]["required"]) <= set(d["parameters"]["properties"])
        json.dumps(d)


@pytest.mark.parametrize("name,args", [
    ("profilo_dataset", {}),
    ("esegui_sql", {"query": "SELECT Regione, COUNT(*) n FROM df GROUP BY 1"}),
    ("aggrega", {"per": ["Categoria"], "metriche": [{"colonna": "Fatturato", "funzione": "sum"}]}),
    ("statistiche", {"colonne": ["Fatturato", "Canale"]}),
    ("correlazioni", {"metodo": "spearman"}),
    ("test_ipotesi", {"test": "anova", "variabile1": "Fatturato", "variabile2": "Regione"}),
    ("outlier", {"metodo": "zscore"}),
    ("serie_temporale", {"colonna_data": "Data Ordine", "colonna_valore": "Fatturato"}),
    ("previsione", {"colonna_data": "Data Ordine", "colonna_valore": "Fatturato", "orizzonte": 2}),
])
def test_every_tool_runs_and_is_serializable(ctx, sales, name, args):
    before = sales.copy()
    res = T.run_tool(name, args, ctx)
    assert not res.error, res.error
    assert res.code and res.description
    json.dumps(res.payload)                       # il risultato deve poter andare al modello
    assert sales.equals(before)                   # gli strumenti sono di sola lettura


def test_aggregate_values(ctx, sales):
    res = T.run_tool("aggrega", {"per": ["Regione"], "ordina_per": "Fatturato_sum", "limite": 1,
                                 "metriche": [{"colonna": "Fatturato", "funzione": "sum"}]}, ctx)
    rows = res.payload["risultato"]["righe"]
    assert rows[0][0] == "Nord" and len(rows) == 1 and res.payload["gruppi_totali"] == 5
    assert rows[0][1] == pytest.approx(sales.loc[sales["Regione"] == "Nord", "Fatturato"].sum(), abs=0.01)
    assert len(res.table) == 5                    # il risultato completo resta in locale


def test_tables_are_truncated_for_the_model(ctx, sales):
    res = T.run_tool("esegui_sql", {"query": "SELECT * FROM df"}, ctx)
    assert len(res.payload["risultato"]["righe"]) == T.MAX_ROWS_TO_MODEL
    assert res.payload["risultato"]["righe_totali"] == len(sales) and len(res.table) == len(sales)


def test_errors_go_back_to_the_model(ctx):
    assert "non esiste" in T.run_tool("aggrega", {"per": ["Inesistente"]}, ctx).payload["errore"]
    assert "inesistente" in T.run_tool("cancella_tutto", {}, ctx).payload["errore"]
    assert T.run_tool("esegui_sql", {"query": "SELECT * FROM read_csv('x.csv')"}, ctx).error
    assert T.run_tool("aggrega", {"per": ["Regione"],
                                  "metriche": [{"colonna": "Fatturato", "funzione": "drop"}]}, ctx).error
    assert T.run_tool("test_ipotesi", {"test": "magia", "variabile1": "Fatturato"}, ctx).error


def test_unknown_arguments_are_ignored(ctx):
    assert not T.run_tool("profilo_dataset", {"inventato": 1}, ctx).error


def test_column_names_are_case_insensitive(ctx, sales):
    assert T.resolve(sales, "fatturato") == "Fatturato"


def test_clean_value():
    import numpy as np
    import pandas as pd

    assert T.clean_value(np.float64("nan")) is None
    assert T.clean_value(np.int64(3)) == 3 and isinstance(T.clean_value(np.int64(3)), int)
    assert T.clean_value(pd.Timestamp("2025-03-01")) == "2025-03-01"
    assert T.clean_value(1234.56789) == 1234.57
    assert T.clean_value(0.000123456789) == pytest.approx(0.000123457)
