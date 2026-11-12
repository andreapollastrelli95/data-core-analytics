"""Interfaccia: l'app si avvia e la scheda «Analisi guidata» completa una catena simulata.

Richiede Tk e uno schermo (su Linux: xvfb-run). Senza, i test vengono saltati.
"""
import os
import time
from types import SimpleNamespace

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")

from conftest import SAMPLE  # noqa: E402


@pytest.fixture(scope="module")
def window():
    """Una sola finestra per tutti i test, come nell'uso reale."""
    try:
        probe = tk.Tk()
        probe.destroy()
    except tk.TclError:
        pytest.skip("nessuno schermo disponibile")
    import main

    root = main.App()
    root.errors = []
    root.show_error = lambda exc: root.errors.append(exc)
    root.info = lambda message: None
    root.confirm = lambda *args, **kwargs: True
    yield root
    root.destroy()


@pytest.fixture
def app(window):
    import cleaner

    window.errors.clear()
    df, code = cleaner.load_file(SAMPLE)
    window.load_dataset(df, cleaner.summarize(df), SAMPLE, "Caricato", code)
    tab = guided(window)
    tab.run = None
    tab.question.delete("1.0", "end")
    window.update()
    return window


def pump(window, condition, seconds=15):
    end = time.time() + seconds
    while time.time() < end and not condition():
        window.update()
        time.sleep(0.02)
    window.update()
    assert condition(), "operazione non completata in tempo"


def guided(window):
    return next(t for t in window.tabs if type(t).__name__ == "GuidedTab")


def test_app_has_all_tabs(app):
    assert len(app.tabs) == 7 and not app.errors
    app.toggle_ai()
    app.update()
    assert app.ai_visible and not app.errors
    app.toggle_ai()


def test_quality_only_from_the_tab(app):
    tab = guided(app)
    tab._quality_only()
    pump(app, lambda: tab.run is not None)
    assert tab.run.ai_used is False and not app.errors
    assert "duplicate" in tab.out.get("1.0", "end")
    assert tab.btn_html.cget("state") == "normal"


def test_guided_run_from_the_tab(app, monkeypatch, tmp_path):
    import ai_engine as ai

    replies = [
        SimpleNamespace(text="", raw=None, tokens=1, calls=[
            ("aggrega", {"per": ["Regione"], "metriche": [{"colonna": "Fatturato", "funzione": "sum"}]})]),
        SimpleNamespace(text="Il Nord fattura 53.691,52.", raw=None, tokens=1, calls=[]),
        SimpleNamespace(text='{"verdetto": "approvato", "problemi": []}', raw=None, tokens=1, calls=[]),
        SimpleNamespace(text="## Risposta in breve\nIl Nord fattura 53.691,52 €.", raw=None, tokens=1,
                        calls=[])]
    monkeypatch.setattr(ai, "has_api_key", lambda: True)
    monkeypatch.setattr(ai, "generate", lambda messages, system, **kw: replies.pop(0))
    tab = guided(app)
    tab.question.insert("1.0", "Quale regione fattura di più?")
    ops_before = len(app.ops)
    tab._start()
    pump(app, lambda: tab.run is not None and not tab.running())
    assert not app.errors, app.errors
    assert tab.run.verdict["verdetto"] == "approvato" and tab.run.model_calls == 4
    assert "53.691,52" in tab.out.get("1.0", "end")
    assert "aggrega" in tab.steps_box.get("1.0", "end")
    assert "groupby" in tab.code_box.get("1.0", "end")
    assert len(app.ops) == ops_before + 1                      # il calcolo entra nel registro
    assert "Analisi guidata" in app.log_entries[-1]
    monkeypatch.setattr(app, "ask_save", lambda *a, **k: str(tmp_path / "analisi.md"))
    monkeypatch.setattr(os, "startfile", lambda path: None, raising=False)
    tab._export("md")
    assert "Quale regione fattura di più?" in (tmp_path / "analisi.md").read_text(encoding="utf-8")


def test_guided_requires_question_and_key(app, monkeypatch):
    import ai_engine as ai
    from cleaner import DataError

    tab = guided(app)
    with pytest.raises(DataError):
        tab._start()
    monkeypatch.setattr(ai, "has_api_key", lambda: False)
    tab.question.insert("1.0", "Totale?")
    with pytest.raises(DataError, match="Chiave"):
        tab._start()


def test_dictionary_dialog_saves(app):
    import context_store as store
    from gui.tab_guided import DictionaryDialog

    tab = guided(app)
    dialog = DictionaryDialog(app, app.df, app.source, tab.dictionary, tab._refresh_dictionary)
    app.update()
    dialog.entries["Fatturato"].insert(0, "Importo netto")
    dialog.metrics.insert("1.0", "Scontrino medio: Fatturato / righe")
    dialog._save()
    app.update()
    assert store.load_dictionary(app.source)["colonne"] == {"Fatturato": "Importo netto"}
    assert "1 colonna descritta" in tab.dict_status.cget("text")
