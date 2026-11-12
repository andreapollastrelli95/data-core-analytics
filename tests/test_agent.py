"""Catena dell'analisi guidata, provata con un modello simulato (nessuna chiamata di rete)."""
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

import agent_engine as agent
import report_engine as rep
import review_engine as rv
from cleaner import DataError

AGG = ("aggrega", {"per": ["Regione"], "ordina_per": "Fatturato_sum",
                   "metriche": [{"colonna": "Fatturato", "funzione": "sum"}]})
TOTAL = ("esegui_sql", {"query": "SELECT SUM(Fatturato) AS totale FROM df"})
DRAFT = "Il Nord fattura di più: 53.691,52 su 135.016,34 (39,8%)."
VERDICT = ('{"verdetto": "approvato_con_riserve", "problemi": [{"gravita": "media", '
           '"descrizione": "Duplicati inclusi nei totali."}], "correzioni": [], '
           '"limiti": ["Un solo anno di dati."]}')
REPORT = ("## Risposta in breve\nIl Nord fattura di più: 53.691,52 € su 135.016,34 € (39,8%).\n"
          "## Evidenze\n- Centro: 34.218,19 €")


def reply(text="", calls=()):
    return SimpleNamespace(text=text, calls=list(calls), raw=None, tokens=10)


class FakeModel:
    """Restituisce le risposte preparate e registra come è stato chiamato."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, system, **kw):
        self.seen.append({"messages": [dict(m) for m in messages], "system": system, **kw})
        return self.replies.pop(0)


def standard():
    return FakeModel([reply(calls=[AGG, TOTAL]), reply(DRAFT), reply(VERDICT), reply(REPORT)])


def test_full_chain(sales):
    model, events = standard(), []
    run = agent.run_guided(sales, "Quale regione fattura di più?", "vendite.csv", llm=model,
                           on_event=lambda kind, payload: events.append(kind))
    assert [s.phase for s in run.steps] == ["controllo", "analisi", "analisi", "analisi",
                                            "revisione", "report"]
    assert run.model_calls == 4 and run.tokens == 40
    assert run.verdict["verdetto"] == "approvato_con_riserve"
    assert run.grounding["non_trovati"] == [] and "39,8%" in run.grounding["derivati"]
    assert "## Verifiche automatiche" in run.report_md and "Approvato con riserve" in run.report_md
    assert events.count("phase") == 4 and events.count("step") == len(run.steps)
    # il modello riceve la struttura dei dati, non le righe
    first = model.seen[0]["messages"][0]["text"]
    assert "Fatturato" in first and "DOMANDA: Quale regione" in first and "2025-01-02" not in first
    # revisore e report vedono i risultati calcolati e la bozza
    assert "53691.52" in model.seen[2]["messages"][0]["text"] and model.seen[2]["json_mode"] is True
    assert DRAFT in model.seen[3]["messages"][0]["text"]


def test_invented_number_is_caught(sales):
    model = standard()
    model.replies[-1] = reply(REPORT + "\n- Margine stimato: 77.123,40 €")
    run = agent.run_guided(sales, "Quale regione fattura di più?", llm=model)
    assert run.grounding["non_trovati"] == ["77.123,40"]
    assert rv.worst(run.findings) == rv.ERROR and "77.123,40" in run.report_md


def test_call_budget_is_respected(sales):
    """Un modello che continua a chiedere strumenti viene fermato: mai più di 6 chiamate."""
    model = FakeModel([reply(calls=[TOTAL])] * 3 + [reply(DRAFT), reply(VERDICT), reply(REPORT)])
    run = agent.run_guided(sales, "Totale?", llm=model)
    assert run.model_calls == agent.MAX_MODEL_CALLS == 6
    assert model.seen[3]["allow_tools"] is False          # ultimo turno: deve rispondere a parole
    assert "esaurito" in model.seen[3]["messages"][-1]["text"]


def test_tool_errors_are_returned_to_the_model(sales):
    bad = ("aggrega", {"per": ["Regioni"]})
    model = FakeModel([reply(calls=[bad]), reply(calls=[AGG, TOTAL]), reply(DRAFT), reply(VERDICT),
                       reply(REPORT)])
    run = agent.run_guided(sales, "Quale regione fattura di più?", llm=model)
    assert "errore" in model.seen[1]["messages"][-1]["results"][0][1]
    assert [s.failed for s in run.tool_steps()] == [True, False, False]
    assert any("non riuscite" in f.title for f in run.findings)


def test_too_many_tools_in_one_turn(sales):
    model = FakeModel([reply(calls=[TOTAL] * 8), reply(DRAFT), reply(VERDICT), reply(REPORT)])
    run = agent.run_guided(sales, "Totale?", llm=model)
    assert len(run.tool_steps()) == agent.MAX_TOOLS_PER_ROUND
    assert len(model.seen[1]["messages"][-1]["results"]) == 8      # una risposta per ogni chiamata


def test_answer_without_calculations_is_flagged(sales):
    model = FakeModel([reply("Direi il Nord, circa 50.000,00."), reply(VERDICT), reply(REPORT)])
    run = agent.run_guided(sales, "Quale regione fattura di più?", llm=model)
    assert run.findings[0].level == rv.ERROR and "Nessun calcolo" in run.findings[0].title


def test_unreadable_reviewer_does_not_block(sales):
    model = standard()
    model.replies[2] = reply("Mi sembra tutto a posto!")
    run = agent.run_guided(sales, "Quale regione fattura di più?", llm=model)
    assert run.verdict["verdetto"] == "approvato_con_riserve"
    assert "non ha restituito" in run.verdict["problemi"][0]["descrizione"]


def test_parse_verdict_variants():
    v = agent.parse_verdict('```json\n{"verdetto": "RESPINTO", "problemi": ["totale sbagliato"]}\n```')
    assert v["verdetto"] == "respinto" and v["problemi"][0]["gravita"] == "media"
    assert agent.parse_verdict('{"verdetto": "boh"}')["verdetto"] == "approvato_con_riserve"


def test_dictionary_and_secondary_reach_the_model(sales):
    model = standard()
    dictionary = {"descrizione": "", "colonne": {"Fatturato": "Importo netto dopo lo sconto"},
                  "metriche": [], "regole": ["Escludi i resi"]}
    agent.run_guided(sales, "Quale regione fattura di più?", llm=model, dictionary=dictionary,
                     df2=pd.DataFrame({"Regione": ["Nord"], "Area": ["A"]}), sample_rows=3)
    first = model.seen[0]["messages"][0]["text"]
    assert "Importo netto dopo lo sconto" in first and "Escludi i resi" in first
    assert "df2" in first and "2025-01-02" in first                 # righe di esempio su richiesta
    assert "Escludi i resi" in model.seen[2]["messages"][0]["text"]


def test_interrupt_and_validation(sales):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(agent.Interrupted):
        agent.run_guided(sales, "Totale?", llm=standard(), cancel=cancel)
    with pytest.raises(DataError):
        agent.run_guided(sales, "   ", llm=standard())
    with pytest.raises(DataError):
        agent.run_guided(sales, "Totale?", llm=FakeModel([reply("")] * 4))


def test_quality_check_finds_seeded_problems():
    df = pd.DataFrame({
        "citta": ["Roma", "roma ", "Milano", "Milano", "Bari", "Bari"] * 10,
        "prezzo": ["1,5", "2", "3", "4", "5", "6"] * 10,
        "importo": [10.0, 12.0, 11.0, 9.0, 10.0, -500.0] + [10.0] * 54,
        "vuota": [None] * 60, "fissa": ["x"] * 60})
    titles = " | ".join(f.title for f in agent.quality_check(df)["issues"])
    for expected in ("righe duplicate", "scritti in più modi", "numeri salvati come testo",
                     "completamente vuote", "un solo valore", "valori negativi"):
        assert expected in titles, expected


def test_quality_only_needs_no_model(sales):
    run = agent.run_quality_only(sales, "vendite.csv")
    assert run.ai_used is False and run.model_calls == 0
    assert "12 righe duplicate" in run.report_md and "senza inviare dati" in run.report_md


def test_clean_dataset_passes_quality():
    df = pd.DataFrame({"a": range(50), "b": [f"k{i}" for i in range(50)]})
    assert [f.level for f in agent.quality_check(df)["issues"]] == [rv.OK]


def test_script_and_html_export(sales, tmp_path):
    run = agent.run_guided(sales, "Quale regione fattura di più?", "vendite.csv", llm=standard())
    script = agent.run_script(run)
    scope = {"df": sales}
    exec(compile(script, "analisi.py", "exec"), scope)  # noqa: S102 - codice generato dall'app
    assert float(scope["risultato"]["totale"][0]) == pytest.approx(sales["Fatturato"].sum())
    path = tmp_path / "analisi.html"
    rep.guided_html(str(path), run)
    html = path.read_text(encoding="utf-8")
    assert "Quale regione fattura di più?" in html and "Verifiche automatiche" in html
    assert "Appendice: passi eseguiti" in html and "duckdb.connect" in html
