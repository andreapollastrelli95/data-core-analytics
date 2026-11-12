import os

import pytest

import context_store as store
from cleaner import DataError


def test_dictionary_roundtrip(sales):
    source = r"C:\dati\Vendite Esempio.csv"
    assert store.is_empty(store.load_dictionary(source))
    assert store.dictionary_prompt(store.load_dictionary(source), sales) == ""
    path = store.save_dictionary(source, {
        "descrizione": "Vendite 2025", "colonne": {"Fatturato": "Importo netto", "Canale": "  "},
        "metriche": store.parse_metrics("Scontrino medio: Fatturato / righe"),
        "regole": store.parse_rules("- I duplicati sono errori\n\n")})
    assert os.path.isfile(path) and path.startswith(os.environ["DCA_HOME"])
    d = store.load_dictionary(source)
    assert d["colonne"] == {"Fatturato": "Importo netto"}            # le voci vuote non si salvano
    assert d["metriche"] == [{"nome": "Scontrino medio", "definizione": "Fatturato / righe"}]
    assert d["regole"] == ["I duplicati sono errori"]
    prompt = store.dictionary_prompt(d, sales)
    assert "Importo netto" in prompt and "Scontrino medio" in prompt and "Vendite 2025" in prompt


def test_dictionary_follows_the_file_name():
    assert store.dataset_key("C:/a/Vendite 2025.xlsx") == store.dataset_key("D:/b/vendite 2025.csv")


def test_prompt_skips_columns_no_longer_present(sales):
    d = {"descrizione": "", "colonne": {"Fatturato": "x", "Sparita": "y"}, "metriche": [], "regole": []}
    assert "Sparita" not in store.dictionary_prompt(d, sales)


def test_metric_format_is_validated():
    with pytest.raises(DataError):
        store.parse_metrics("senza definizione")


def test_procedures_builtin_and_saved():
    names = [p["nome"] for p in store.list_procedures()]
    assert "Report di sintesi" in names and not any(p["utente"] for p in store.list_procedures())
    store.save_procedure("Regioni", "Quale regione fattura di più?")
    store.save_procedure("regioni", "Quale regione fattura di meno?")    # stesso nome: sostituisce
    saved = [p for p in store.list_procedures() if p["utente"]]
    assert len(saved) == 1 and saved[0]["domanda"].endswith("meno?")
    assert store.get_procedure("regioni")["utente"] is True
    store.delete_procedure("regioni")
    assert not [p for p in store.list_procedures() if p["utente"]]


def test_procedure_validation():
    for name, question in (("", "x"), ("x", ""), ("Report di sintesi", "x")):
        with pytest.raises(DataError):
            store.save_procedure(name, question)
    with pytest.raises(DataError):
        store.delete_procedure("Report di sintesi")
    with pytest.raises(DataError):
        store.delete_procedure("mai salvata")


def test_corrupted_file_is_reported(tmp_path):
    os.makedirs(os.path.dirname(store.procedures_path()), exist_ok=True)
    with open(store.procedures_path(), "w", encoding="utf-8") as fh:
        fh.write("{non json")
    with pytest.raises(DataError):
        store.list_procedures()
