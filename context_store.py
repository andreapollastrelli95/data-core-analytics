"""Contesto riutilizzabile: dizionario dati per dataset e procedure di analisi salvate.

Tutto è salvato in file JSON nella cartella dell'utente (~/.data_core_analytics, oppure quella
indicata dalla variabile d'ambiente DCA_HOME), quindi resta disponibile tra una sessione e l'altra
e non finisce mai nel repository. Il modulo non dipende dall'interfaccia grafica.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import pandas as pd

from cleaner import DataError

HOME_ENV = "DCA_HOME"


def home() -> str:
    return os.environ.get(HOME_ENV) or os.path.join(os.path.expanduser("~"), ".data_core_analytics")


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "dataset"


def _read(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise DataError(f"Impossibile leggere '{path}':\n{exc}") from exc


def _write(path: str, data) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        raise DataError(f"Impossibile salvare '{path}':\n{exc}") from exc


# --------------------------------------------------------------------------- #
# Dizionario dati
# --------------------------------------------------------------------------- #
def dataset_key(source: str) -> str:
    """Chiave del dizionario: il nome del file senza estensione (vale per ogni sua versione)."""
    return _slug(os.path.splitext(os.path.basename(source or "dataset"))[0])


def dictionary_path(source: str) -> str:
    return os.path.join(home(), "dizionari", dataset_key(source) + ".json")


def empty_dictionary(source: str = "") -> dict:
    return {"dataset": os.path.basename(source or ""), "descrizione": "", "colonne": {},
            "metriche": [], "regole": []}


def load_dictionary(source: str) -> dict:
    data = _read(dictionary_path(source), None)
    base = empty_dictionary(source)
    if isinstance(data, dict):
        base["descrizione"] = str(data.get("descrizione", "") or "")
        if isinstance(data.get("colonne"), dict):
            base["colonne"] = {str(k): str(v) for k, v in data["colonne"].items() if str(v).strip()}
        base["metriche"] = [{"nome": str(m.get("nome", "")).strip(),
                             "definizione": str(m.get("definizione", "")).strip()}
                            for m in data.get("metriche", []) if isinstance(m, dict)
                            and str(m.get("nome", "")).strip()]
        base["regole"] = [str(r).strip() for r in data.get("regole", []) if str(r).strip()]
    return base


def save_dictionary(source: str, dictionary: dict) -> str:
    path = dictionary_path(source)
    clean = empty_dictionary(source)
    clean["descrizione"] = str(dictionary.get("descrizione", "")).strip()
    clean["colonne"] = {str(k): str(v).strip() for k, v in dictionary.get("colonne", {}).items()
                        if str(v).strip()}
    clean["metriche"] = [m for m in dictionary.get("metriche", []) if m.get("nome")]
    clean["regole"] = [r for r in dictionary.get("regole", []) if str(r).strip()]
    _write(path, clean)
    return path


def is_empty(dictionary: dict | None) -> bool:
    return not dictionary or not (dictionary.get("descrizione") or dictionary.get("colonne")
                                  or dictionary.get("metriche") or dictionary.get("regole"))


def parse_metrics(text: str) -> list[dict]:
    """Una metrica per riga, nel formato «nome: definizione»."""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•").strip()
        if not line:
            continue
        name, sep, definition = line.partition(":")
        if not sep or not name.strip() or not definition.strip():
            raise DataError(f"Metrica non valida: «{line}».\n\nScrivi una metrica per riga nel "
                            "formato «nome: definizione», ad esempio\n"
                            "«Scontrino medio: Fatturato diviso numero di ordini».")
        out.append({"nome": name.strip(), "definizione": definition.strip()})
    return out


def format_metrics(metrics: list[dict]) -> str:
    return "\n".join(f"{m['nome']}: {m['definizione']}" for m in metrics)


def parse_rules(text: str) -> list[str]:
    return [line.strip().lstrip("-•").strip() for line in text.splitlines()
            if line.strip().lstrip("-•").strip()]


def dictionary_prompt(dictionary: dict | None, df: pd.DataFrame | None = None) -> str:
    """Testo del dizionario da dare al modello (vuoto se il dizionario non è compilato)."""
    if is_empty(dictionary):
        return ""
    lines = ["DIZIONARIO DATI (definizioni fornite dall'utente: hanno la precedenza sulle tue ipotesi)"]
    if dictionary.get("descrizione"):
        lines += ["Descrizione del dataset: " + dictionary["descrizione"]]
    cols = dictionary.get("colonne", {})
    present = {str(c) for c in df.columns} if df is not None else None
    shown = [(c, d) for c, d in cols.items() if present is None or c in present]
    if shown:
        lines += ["Significato delle colonne:"] + [f"- {c}: {d}" for c, d in shown]
    if dictionary.get("metriche"):
        lines += ["Definizione delle metriche:"] + [f"- {m['nome']}: {m['definizione']}"
                                                    for m in dictionary["metriche"]]
    if dictionary.get("regole"):
        lines += ["Regole di business:"] + [f"- {r}" for r in dictionary["regole"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Procedure salvate
# --------------------------------------------------------------------------- #
BUILTIN = [
    {"nome": "Report di sintesi",
     "domanda": "Prepara una sintesi del dataset per chi deve decidere: quali sono i numeri "
                "principali, come si distribuiscono tra i segmenti più importanti e quali sono le "
                "tre cose da sapere."},
    {"nome": "Andamento nel tempo",
     "domanda": "Come è cambiata nel tempo la metrica principale? Individua trend, periodi "
                "anomali e stagionalità, e stima i prossimi tre periodi con l'errore atteso."},
    {"nome": "Confronto tra segmenti",
     "domanda": "Confronta i segmenti (le principali variabili categoriche) sulla metrica "
                "principale: chi va meglio, chi va peggio e se le differenze sono statisticamente "
                "significative."},
    {"nome": "Cosa guida la metrica",
     "domanda": "Quali variabili sono più legate alla metrica principale? Usa correlazioni e "
                "confronti tra gruppi e distingui i legami forti da quelli deboli."},
    {"nome": "Valori anomali",
     "domanda": "Ci sono valori anomali o righe sospette? Quantificali, indica dove si "
                "concentrano e quanto spostano i totali."},
]


def procedures_path() -> str:
    return os.path.join(home(), "procedure.json")


def _saved() -> list[dict]:
    data = _read(procedures_path(), [])
    return [{"nome": str(p["nome"]).strip(), "domanda": str(p["domanda"]).strip()}
            for p in data if isinstance(p, dict) and str(p.get("nome", "")).strip()
            and str(p.get("domanda", "")).strip()] if isinstance(data, list) else []


def list_procedures() -> list[dict]:
    """Procedure predefinite seguite da quelle salvate dall'utente (campo «utente»: True)."""
    return ([{**p, "utente": False} for p in BUILTIN]
            + [{**p, "utente": True} for p in _saved()])


def get_procedure(name: str) -> dict | None:
    return next((p for p in list_procedures() if p["nome"] == name), None)


def save_procedure(name: str, question: str) -> None:
    name, question = name.strip(), question.strip()
    if not name:
        raise DataError("Dai un nome alla procedura.")
    if not question:
        raise DataError("Scrivi la domanda da salvare nella procedura.")
    if any(p["nome"].lower() == name.lower() for p in BUILTIN):
        raise DataError(f"«{name}» è una procedura predefinita: scegli un altro nome.")
    saved = [p for p in _saved() if p["nome"].lower() != name.lower()]
    saved.append({"nome": name, "domanda": question})
    _write(procedures_path(), saved)


def delete_procedure(name: str) -> None:
    if any(p["nome"] == name for p in BUILTIN):
        raise DataError("Le procedure predefinite non si possono eliminare.")
    saved = _saved()
    kept = [p for p in saved if p["nome"] != name]
    if len(kept) == len(saved):
        raise DataError(f"Procedura «{name}» non trovata.")
    _write(procedures_path(), kept)
